import argparse
import json
import os
import random
import wandb
import pickle
import time
import glob
from pathlib import Path
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from data_utils import (WOSDataset, get_examples_from_dialogues, load_dataset,
                        set_seed)
from eval_utils import DSTEvaluator, eval_wrong_count
from evaluation import _evaluation
from inference import inference_trade, save_trade
from model import TRADE, masked_cross_entropy_for_value, SUMBT
from preprocessor import TRADEPreprocessor, TRADEPreprocessorTest
from torch.cuda.amp import autocast,  GradScaler
from pprint import pprint
import torch.cuda.amp as amp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.
    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(args):
    # random seed 고정
    set_seed(args.random_seed)

    save = False
    if args.wandb_name:
        save_dir = f"{args.model_dir}/{args.wandb_name}"
        save = True
        save_dir = increment_path(save_dir)

    # Define Preprocessor
    added_token_num = 0
    slot_meta = json.load(open("/opt/ml/input/data/train_dataset/slot_meta.json"))
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    # added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[STATE]"]})
    args.vocab_size = added_token_num + tokenizer.vocab_size
    processor = TRADEPreprocessor(slot_meta, tokenizer)
    args.n_gate = len(processor.gating2id)



    if not os.path.exists(os.path.join(args.data_dir, "train_features.bin")):

        train_data_file = "/opt/ml/input/data/train_dataset/train_dials.json"
        train_data, dev_data, dev_labels = load_dataset(train_data_file)

        train_examples = get_examples_from_dialogues(train_data,
                                                     user_first=False,
                                                     dialogue_level=False)
        dev_examples = get_examples_from_dialogues(dev_data,
                                                   user_first=False,
                                                   dialogue_level=False)

        train_features = processor.convert_examples_to_features(train_examples)
        dev_features = processor.convert_examples_to_features(dev_examples)

        with open('trade_data/train_features.bin', 'wb') as f:
            pickle.dump(train_features, f)
        with open('trade_data/dev_features.bin', 'wb') as f:
            pickle.dump(dev_features, f)
        with open('trade_data/dev_labels.bin', 'wb') as f:
            pickle.dump(dev_labels, f)
    else:
        with open('trade_data/train_features.bin', 'rb') as f:
            train_features = pickle.load(f)
        with open('trade_data/dev_features.bin', 'rb') as f:
            dev_features = pickle.load(f)
        with open('trade_data/dev_labels.bin', 'rb') as f:
            dev_labels = pickle.load(f)


    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    # Model 선언
    model = TRADE(args, tokenized_slot_meta, slot_meta)
    model.to(device)
    print("Model is initialized")

    if save:
        wandb.init(project='pstage3', entity='ekzm8523')
        wandb.config.update(args)
        wandb.run.name = args.wandb_name
        wandb.watch(model)


    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
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
    predictions = inference_trade(model, dev_loader, processor, device)

    # check dev dataset
    for data in dev_data:
        if not data.guid in dev_labels:
            raise Exception("wrong dev data set check of the dataset")


    print("# dev:", len(dev_data))

    # Optimizer 및 Scheduler 선언

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    n_epochs = args.epochs
    t_total = len(train_loader) * n_epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    if save:
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        if not os.path.exists(f"{save_dir}"):
            os.mkdir(f"{save_dir}")
        json.dump(
            vars(args),
            open(f"{save_dir}/exp_config.json", "w"),
            indent=2,
            ensure_ascii=False,
        )

        json.dump(
            slot_meta,
            open(f"{save_dir}/slot_meta.json", "w"),
            indent=2,
            ensure_ascii=False,
        )

    scaler = GradScaler(enabled=use_amp)
    idx = 0
    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        start = time.time()
        model.train()
        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
                b.to(device) if not isinstance(b, list) else b for b in batch
            ]

            # teacher forcing
            if args.teacher_forcing_ratio > 0.0 and random.random() < args.teacher_forcing_ratio:
                tf = target_ids
            else:
                tf = None
            with amp.autocast(enabled=use_amp):
                all_point_outputs, all_gate_outputs = model(input_ids, segment_ids, input_masks, target_ids.size(-1), tf)

           # generation loss
            loss_1 = loss_fnc_1(
                all_point_outputs.contiguous(),
                target_ids.contiguous().view(-1), # flatten
                tokenizer.pad_token_id,
            )

            # gating loss
            loss_2 = loss_fnc_2(
                all_gate_outputs.contiguous().view(-1, args.n_gate),
                gating_ids.contiguous().view(-1),
            )
            loss = loss_1 + loss_2

            scaler.scale(loss).backward()
            scheduler.step()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()} time: {int(time.time() - start)}second"
                )

        predictions = inference_trade(model, dev_loader, processor, device)
        wrong_value, wrong_slot = eval_wrong_count(predictions, dev_labels)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")

        # 걸린 시간 계산
        epoch_time = int(time.time() - start)
        h = epoch_time // 3600
        epoch_time -= h * 3600
        m = epoch_time // 60
        s = epoch_time - m * 60
        print(f" 걸린 시간 : {h}시 {m}분 {s}초")
        print("---------------wrong value top 10-----------------")
        pprint(wrong_value)
        print("---------------wrong slot top 10------------------")
        pprint(wrong_slot)
        print("-"*50)
        if args.wandb_name:
            wandb.log({
                "loss": loss.item(),
                "gen_loss": loss_1.item(),
                "gate_loss": loss_2.item(),
                "joint_acc": eval_result['joint_goal_accuracy'],
                "turn_slot_acc": eval_result['turn_slot_accuracy'],
                "turn_slot_f1": eval_result['turn_slot_f1'],
            })
        if best_score < eval_result['joint_goal_accuracy']:
            idx = (idx + 1) % 3
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            if save:
                torch.save(model.state_dict(), f"{save_dir}/best_model{idx}.bin")
                save_info = {"model_name": f"best_model{idx}.bin", "epoch": epoch, "JGA": best_score}
                json.dump(save_info, open(f"{save_dir}/best_model{idx}.json", "w"), indent=2, ensure_ascii=False)
            if save and best_score > 70.0:
                save_trade(model, processor, device, save_dir, epoch)

    print(f"Best checkpoint: {save_dir}/best_model{idx}.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/code/ekzm8523/trade_data")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/output/")
    parser.add_argument("--model", type=str, default="trade", help="select trade or sumbt")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=33)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default="dsksd/bert-ko-small-minimal",
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="GRU의 hidden size", default=768)
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
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()
    print(args)
    train(args)



