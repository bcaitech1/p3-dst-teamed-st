import json
import torch
import wandb
import argparse
import torch.nn as nn

from transformers import BertTokenizer
from data_utils import get_examples_from_dialogues, load_dataset, tokenize_ontology
from preprocessor import SUMBTPreprocessor
from models import SUMBT
from data_utils import WOSDataset, set_seed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from evaluation import _evaluation
from inference_sumbt import inference

n_gpu = 1 if torch.cuda.device_count() < 2 else torch.cuda.device_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    wandb.init(project="SUMBT")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="SUMBT")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/code/train_results")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="dsksd/bert-ko-small-minimal",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=300)
    parser.add_argument(
        "--num_rnn_layers",
        type=int,
        help="rnn layer의 개수",
        default=1,
    )
    parser.add_argument("--zero_init_rnn", type=bool, default=False)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="Sequence 최대 길이",
        default=64,
    )
    parser.add_argument("--max_label_length", type=int, default=12)
    parser.add_argument("--attention_head", type=int, default=4)
    parser.add_argument("--fix_utterance_encoder", type=bool, default=False)
    parser.add_argument("--distance_metric", type=str, default='euclidean')

    args = parser.parse_args()

    wandb.config.update(args)
    wandb.run.name = f"{args.run_name}-{wandb.run.id}"
    wandb.run.save()
    # random seed 고정
    set_seed(args.random_seed)

    # Data Loading
    train_data_file = "/opt/ml/input/data/train/train_dials.json"
    slot_meta = json.load(open("/opt/ml/input/data/train/slot_meta.json"))
    ontology = json.load(open("/opt/ml/input/data/train/ontology.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(data=train_data,
                                                 user_first=True,
                                                 dialogue_level=True)

    dev_examples = get_examples_from_dialogues(data=dev_data,
                                               user_first=True,
                                               dialogue_level=True)
    #max turn 정의
    max_turn = max([len(e['dialogue']) for e in train_data])

    # Preprocessor 정의
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    processor = SUMBTPreprocessor(slot_meta,
                                  tokenizer,
                                  ontology=ontology,  # predefined ontology
                                  max_seq_length=64,  # 각 turn마다 최대 길이
                                  max_turn_length=max_turn)  # 각 dialogue의 최대 turn 길이
    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, 12)
    num_labels = [len(s) for s in slot_values_ids]  # 각 Slot 별 후보 Values의 갯수

    print("Tokenized Slot: ", slot_type_ids.size())
    for slot, slot_value_id in zip(slot_meta, slot_values_ids):
        print(f"Tokenized Value of {slot}", slot_value_id.size())


    print(f"Subword Embeddings is loaded from {args.model_name_or_path}")
    model = SUMBT(args, num_labels, device)
    model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV
    model.to(device)
    print("Model is initialized")

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=8, sampler=train_sampler, collate_fn=processor.collate_fn)

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(dev_data, batch_size=8, sampler=dev_sampler, collate_fn=processor.collate_fn)
    print("# dev:", len(dev_data))

    # Optimizer 및 Scheduler 선언
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    t_total = len(train_loader) * args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_ratio), num_training_steps=t_total
    )

    best_score, best_checkpoint = 0, 0
    for epoch in range(args.num_train_epochs):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, segment_ids, input_masks, target_ids, num_turns, guids = \
                [b.to(device) if not isinstance(b, list) else b for b in batch]

            # Forward
            if n_gpu == 1:
                loss, loss_slot, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
            else:
                loss, _, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)

            batch_loss.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print('[%d/%d] [%d/%d] %f' % (epoch, args.num_train_epochs, step, len(train_loader), loss.item()))

        predictions = inference(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")
            wandb.log({k: v})
        if best_score < eval_result["joint_goal_accuracy"]:
            print("Update Best checkpoint!")
            best_score = eval_result["joint_goal_accuracy"]
            best_checkpoint = epoch

        torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
