import argparse
import json
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from inference import direct_output
from data_utils import WOSDataset, get_examples_from_dialogues, load_dataset, set_seed
from evaluation import _evaluation
from inference import somdst_inference, increment_path
from model.somdst import SOMDST
from preprocessor import SOMDSTPreprocessor

import torch.cuda.amp as amp
from loss import masked_cross_entropy_for_value

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="SOMDST")
    parser.add_argument("--data_dir", type=str, default="/opt/ml/somdst_data/")
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--model_name_or_path", type=str, default="dsksd/bert-ko-small-minimal",)

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int, default=None,)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    save = False
    if args.save_dir:
        save = True
        save_dir = increment_path(args.save_dir)

    set_seed(args.random_seed)

    # Data Loading
    slot_meta = json.load(open("/opt/ml/input/DST_data/train_dataset/slot_meta.json"))
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    added_token_num = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[SLOT]", "[NULL]", "[EOS]"]}
    )
    # Define Preprocessor
    print(f"preprocessing data !! ")
    processor = SOMDSTPreprocessor(slot_meta, tokenizer, max_seq_length=args.max_seq_length)
    args.vocab_size = tokenizer.vocab_size + added_token_num

    train_data_file = "/opt/ml/input/DST_data/train_dataset/train_dials.json"
    train_data, dev_data, dev_labels = load_dataset(train_data_file)
    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    # Extracting Featrues
    train_features = processor.convert_examples_to_features(train_examples)

    # Model 선언
    model = SOMDST(args, 5, 4, processor.op2id["update"])
    model.to(device)
    print("Model is initialized")

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
        num_workers=4,
    )
    print("# train:", len(train_data))
    print("# dev:", len(dev_examples))

    # Optimizer 및 Scheduler 선언
    n_epochs = args.epochs
    t_total = len(train_loader) * n_epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        json.dump(
            vars(args),
            open(f"{save_dir}/exp_config.json", "w"),
            indent=2,
            ensure_ascii=False,
        )

    idx = 0
    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):

        model.train()
        for step, batch in enumerate(tqdm(train_loader)):

            batch = [
                b.to(device)
                if not isinstance(b, int) and not isinstance(b, list)
                else b
                for b in batch
            ]
            (
                input_ids,
                input_masks,
                segment_ids,
                slot_position_ids,
                gating_ids,
                domain_ids,
                target_ids,
                max_update,
                max_value,
                guids,
            ) = batch

            # teacher forcing
            if (
                    args.teacher_forcing_ratio > 0.0
                    and random.random() < args.teacher_forcing_ratio
            ):
                tf = target_ids
            else:
                tf = None

            with amp.autocast():

                domain_scores, state_scores, gen_scores = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    slot_positions=slot_position_ids,
                    attention_mask=input_masks,
                    max_value=max_value,
                    op_ids=gating_ids,
                    max_update=max_update,
                    teacher=tf,
                )

                # generation loss
                loss_1 = loss_fnc_1(
                    gen_scores.contiguous(),
                    target_ids.contiguous(),
                    tokenizer.pad_token_id,
                )

                # gating loss
                loss_2 = loss_fnc_2(
                    state_scores.contiguous().view(-1, 4),
                    gating_ids.contiguous().view(-1),
                )
                loss_3 = loss_fnc_2(domain_scores.view(-1, 5), domain_ids.view(-1))
                loss = loss_1 + loss_2 + loss_3

                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}, domain: {loss_3.item()}"
                )

        predictions = somdst_inference(model, dev_examples, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        # 모델은 최대 세개만 저장되도록 설정
        if best_score < eval_result["joint_goal_accuracy"]:
            print("Update Best checkpoint!")
            best_score = eval_result["joint_goal_accuracy"]
            best_checkpoint = epoch

            if save:
                idx = (idx + 1) % 3
                torch.save(model.state_dict(), f"{save_dir}/best_model{idx}.bin")
                save_info = {"model_name": f"best_model{idx}.bin", "epoch": epoch, "JGA": best_score}
                json.dump(save_info, open(f"{save_dir}/best_model{idx}.json", "w"), indent=2, ensure_ascii=False)
    if save:
        torch.save(model.state_dict(), f"{save_dir}/last_model.bin")
        print(f"Best checkpoint: {save_dir}/model-{best_checkpoint}.bin")
        direct_output(save_dir, model, processor)
