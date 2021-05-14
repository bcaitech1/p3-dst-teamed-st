import argparse
import json
import os
import random
import pickle
import glob
from pathlib import Path
import re
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from pytorch_transformers import WarmupLinearSchedule
from data_utils import WOSDataset, get_examples_from_dialogues, load_dataset, set_seed
from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference_somdst import inference
from models import SOMDST, masked_cross_entropy_for_value
from preprocessor import SOMDSTPreprocessor

import torch.cuda.amp as amp
# import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

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


if __name__ == "__main__":
    # wandb.init(project="Stage2-DST")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="SOMDST")

    parser.add_argument(
        "--data_dir", type=str,
        default="/opt/ml/input/data/train_dataset",
        # default="../../input/data/train_dataset",
    )
    parser.add_argument("--model_dir", type=str, default="/opt/ml/result")
    parser.add_argument("--model_name", type=str, default="SOMDST")
    parser.add_argument("--ckpt", type=int, default=48)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=512)
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
    parser.add_argument(
        "--proj_dim",
        type=int,
        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.",
        default=None,
    )
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    args = parser.parse_args()

    # args.data_dir = os.environ["SM_CHANNEL_TRAIN"]
    if args.model_name:
        args.model_dir = os.path.join(args.model_dir, args.model_name)
    else:
        args.model_dir = increment_path(os.path.join(args.model_dir, args.run_name))
    print(args.model_dir)
    # wandb.config.update(args)
    # wandb.run.name = f"{args.run_name}-{wandb.run.id}"
    # wandb.run.save()
    # random seed 고정
    set_seed(args.random_seed)

    # Data Loading
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    # slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json", 'rt', encoding='UTF8'))
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    added_token_num = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[SLOT]", "[NULL]", "[EOS]"]}
    )
    # Define Preprocessor
    processor = SOMDSTPreprocessor(
        slot_meta, tokenizer, max_seq_length=args.max_seq_length
    )
    args.vocab_size = tokenizer.vocab_size + added_token_num
    # args.n_gate = len(processor.gating2id)  # gating 갯수 none, dontcare, ptr
    train_data_file = f"{args.data_dir}/train_dials.json"
    train_data, dev_data, dev_labels = load_dataset(train_data_file)
    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )

    if not os.path.exists(os.path.join(args.data_dir, "train_somdst_features6.pkl")):
        print("Cached Input Features not Found.\nLoad data and save.")

        # Extracting Featrues
        train_features = processor.convert_examples_to_features(train_examples)
        print("Save Data")
        with open(os.path.join(args.data_dir, "train_somdst_features6.pkl"), "wb") as f:
            pickle.dump(train_features, f)
        with open(os.path.join(args.data_dir, "dev_somdst_examples6.pkl"), "wb") as f:
            pickle.dump(dev_examples, f)
        with open(os.path.join(args.data_dir, "dev_somdst_labels6.pkl"), "wb") as f:
            pickle.dump(dev_labels, f)
    else:
        print("Cached Input Features Found.\nLoad data from Cached")
        with open(os.path.join(args.data_dir, "train_somdst_features6.pkl"), "rb") as f:
            train_features = pickle.load(f)
        with open(os.path.join(args.data_dir, "dev_somdst_examples6.pkl"), "rb") as f:
            dev_examples = pickle.load(f)
        with open(os.path.join(args.data_dir, "dev_somdst_labels6.pkl"), "rb") as f:
            dev_labels = pickle.load(f)

    # Model 선언
    model = SOMDST(args, 5, 6, processor.op2id["update"])

    if args.model_name:
        print("Checkpoint Load")
        ckpt = torch.load(os.path.join(args.model_dir, f"model-{args.ckpt}.bin"))
        model.load_state_dict(ckpt)

    # model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화
    # wandb.watch(model)
    # print(f"Subword Embeddings is loaded from {args.model_name_or_path}")
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
    n_epochs = args.num_train_epochs
    t_total = len(train_loader) * n_epochs
    # warmup_steps = int(t_total * args.warmup_ratio)
    num_train_steps = int(len(train_data) / args.train_batch_size * n_epochs)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    # )
    scheduler = WarmupLinearSchedule(optimizer, int(num_train_steps * 0.1),
                                         t_total=num_train_steps)

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    json.dump(
        vars(args),
        open(f"{args.model_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{args.model_dir}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    logger = SummaryWriter(log_dir=args.model_dir)
    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        batch_loss = []
        model.train()
        for step, batch in enumerate(train_loader):
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
                    state_scores.contiguous().view(-1, 6),
                    gating_ids.contiguous().view(-1),
                )
                loss_3 = loss_fnc_2(domain_scores.view(-1, 5), domain_ids.view(-1))
                loss = loss_1 + loss_2 + loss_3
                batch_loss.append(loss.item())

                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 50 == 0:

                print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f, gen_loss : %.3f, dom_loss : %.3f" \
                      % (epoch + 1, n_epochs, step,
                         len(train_loader), np.mean(batch_loss),
                         loss_1.item(), loss_2.item(), loss_3.item()))
                logger.add_scalar("Train/loss", loss, epoch * len(train_loader) + step)
                logger.add_scalar("Train/gen_loss", loss_1, epoch * len(train_loader) + step)
                logger.add_scalar("Train/gating_loss", loss_2, epoch * len(train_loader) + step)
                logger.add_scalar("Train/domain_loss", loss_3, epoch * len(train_loader) + step)
                batch_loss = []
        predictions = inference(model, dev_examples, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")
            # wandb.log({k: v})
        if best_score < eval_result["joint_goal_accuracy"]:
            print("Update Best checkpoint!")
            best_score = eval_result["joint_goal_accuracy"]
        best_checkpoint = epoch + args.ckpt

        torch.save(
            model.state_dict(), f"{args.model_dir}/model-{epoch + args.ckpt}.bin"
        )
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
