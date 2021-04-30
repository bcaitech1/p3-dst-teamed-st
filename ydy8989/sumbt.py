import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from data_utils import (WOSDataset, get_examples_from_dialogues, load_dataset,
                        set_seed, tokenize_ontology)
from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference, inference_sumbt
from model import TRADE, masked_cross_entropy_for_value, SUMBT
from preprocessor import TRADEPreprocessor, SUMBTPreprocessor
from torch.cuda.amp import autocast,  GradScaler
use_amp = True
scaler = GradScaler(enabled=use_amp)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    # random seed 고정
    set_seed(args.random_seed)

    # Data Loading
    # list안에 7000개의 dict
    # train_data[0].keys -> dialogue, dialogue_idx, domain
    # Train과 dev 차이는 dev[0]은 dialogue안에 state(label)가 없다.
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))  # 45개의 slot
    ontology = json.load(open(f"{args.data_dir}/ontology.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)  # item별로 분류 6301개 , 699개

    # list안에 dialogue별로 세분화, dict type -> DSTInputExample type , dev는 label이 none
    train_examples = get_examples_from_dialogues(  # item의 dialogue별로 46170개, 5075개
        train_data, user_first=True, dialogue_level=True
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=True, dialogue_level=True
    )

    # Define Preprocessor
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    max_turn = max([len(e['dialogue']) for e in train_data])
    processor = SUMBTPreprocessor(slot_meta,
                                  tokenizer,
                                  ontology=ontology,  # predefined ontology
                                  max_seq_length=64,  # 각 turn마다 최대 길이
                                  max_turn_length=max_turn)  # 각 dialogue의 최대 turn 길이

    # Extracting Featrues
    # OpenVocabDSTFeature [guid, input_id, segment_id, gating_id, target_ids]
    train_features = processor.convert_examples_to_features(train_examples)
    dev_features = processor.convert_examples_to_features(dev_examples)

    # Ontology pre encoding
    slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, 12)
    num_labels = [len(s) for s in slot_values_ids]  # 각 Slot 별 후보 Values의 갯수

    # Model 선언
    n_gpu = 1 if torch.cuda.device_count() < 2 else torch.cuda.device_count()
    print(n_gpu)
    n_epochs = args.num_train_epochs

    model = SUMBT(args, num_labels, device)
    model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)  # Tokenized Ontology의 Pre-encoding using BERT_SV
    model.to(device)
    print("Model is initialized")

    train_data = WOSDataset(train_features)  # feature와 len만 담긴 dataset
    train_sampler = RandomSampler(train_data)  #
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,
    )
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

    t_total = len(train_loader) * n_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * args.warmup_ratio), num_training_steps=t_total
    )

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(f"{args.model_dir}/{args.model}"):
        os.mkdir(f"{args.model_dir}/{args.model}")

    json.dump(
        vars(args),
        open(f"{args.model_dir}/{args.model}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )

    json.dump(
        slot_meta,
        open(f"{args.model_dir}/{args.model}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    scaler = GradScaler(enabled=use_amp)
    best_score, best_checkpoint = 0, 0
    for epoch in tqdm(range(n_epochs)):
        batch_loss = []
        model.train()
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            optimizer.zero_grad()
            input_ids, segment_ids, input_masks, target_ids, num_turns, guids = \
                [b.to(device) if not isinstance(b, list) else b for b in batch]
            with autocast(enabled=use_amp):
                if n_gpu == 1:
                    loss, loss_slot, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
                else:
                    loss, _, acc, acc_slot, _ = model(input_ids, segment_ids, input_masks, target_ids, n_gpu)
            batch_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)

            scale = scaler.get_scale()
            scaler.update()
            step_scheduler = scaler.get_scale() == scale

            if step_scheduler:
                scheduler.step()


            if step % 50 == 0:
                print('[%d/%d] [%d/%d] %f' % (epoch, n_epochs, step, len(train_loader), loss.item()))

        predictions = inference_sumbt(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch
            torch.save(model.state_dict(), f"{args.model_dir}/{args.model}/best_model.bin")

    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model/")
    parser.add_argument("--model", type=str, default="sumbt", help="select trade or sumbt")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--distance_metric", type=str, default="euclidean")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="dsksd/bert-ko-small-minimal",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_dim", type=int, help="GRU의 hidden size", default=300)
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, subword vocab tokenizer에 의해 특정된다",
        default=None,
    )

    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int,
                        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.",
                        default=None)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--fix_utterance_encoder", type=bool, default=False)
    parser.add_argument("--attn_head", type=int, default=4)
    parser.add_argument("--max_label_length", type=int, default=12)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--zero_init_rnn", type=bool, default=False)
    parser.add_argument("--num_rnn_layers", type=int, default=1)

    args = parser.parse_args()
    print(args)
    train(args)