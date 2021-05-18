import json
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration, BertConfig, BertTokenizer
from transformers import PreTrainedTokenizerFast
from data_utils import *
import os
import pickle

from sklearn.model_selection import train_test_split
from model import BertForMultiLabelSequenceClassification
from evaluation import evaluate, metric
from preprocessor import CoCoClassifierPreprocessor


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/opt/ml/model/coco.bin"
    data = json.load(open('/opt/ml/input/data/train_dataset/train_dials.json'))
    slot_meta = json.load(open('/opt/ml/input/data/train_dataset/slot_meta.json'))
    tokenizer = BertTokenizer.from_pretrained("dsksd/bert-ko-small-minimal")

    bert_config = BertConfig.from_pretrained("dsksd/bert-ko-small-minimal", num_labels=len(slot_meta))
    bert_config.model_name_or_path = "dsksd/bert-ko-small-minimal"
    bert_config.num_labels = 45
    model = BertForMultiLabelSequenceClassification.from_pretrained("dsksd/bert-ko-small-minimal", config=bert_config)

    slot2idx = {slot: i for i, slot in enumerate(slot_meta)}
    idx2slot = {i: slot for i, slot in enumerate(slot_meta)}
    processor = CoCoClassifierPreprocessor(slot_meta, tokenizer, bert_config)

    if not os.path.exists("coco_data"):
        os.mkdir("coco_data")

    if not os.path.exists("coco_data/coco_examples.pkl"):
        examples = []
        for dialogue in tqdm(data):
            examples.extend(get_coco_examples_from_dialogue(dialogue))

        features = []
        for example in tqdm(examples):
            features.append(processor.convert_example_to_feature(example, tokenizer))

        with open("coco_data/coco_examples.pkl", "wb") as f:
            pickle.dump(examples, f)
        with open("coco_data/coco_features.pkl", "wb") as f:
            pickle.dump(features, f)

    else:
        with open("coco_data/coco_examples.pkl", "rb") as f:
            examples = pickle.load(f)
        with open("coco_data/coco_features.pkl", "rb") as f:
            features = pickle.load(f)

    train_features, dev_features = train_test_split(features, test_size=0.2)

    train_data = CoCoClassifierDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(train_data, batch_size=32, sampler=train_sampler, collate_fn=processor.collate_fn)

    dev_data = CoCoClassifierDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=32, collate_fn=processor.collate_fn)

    ##########################
    epochs = 10
    batch_size = 32
    lr = 5e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    #########################

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_total = len(train_loader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * warmup_ratio), num_training_steps=t_total
    )
    model.to(device)

    print("Start training!!")
    for epoch in tqdm(range(epochs)):
        tr_loss = 0
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            input_ids, input_masks, segment_ids, target_ids = (b.to(device) for b in batch)
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, labels=target_ids)

            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"[{epoch}/{epochs}][{step}/{len(train_loader)}] loss : {loss}")
        precision, recall = evaluate(model, device, dev_dataloader)
        print("***** Eval on dev set *****")
        print("Current precision = %.4f" % (precision))
        print("Current recall = %.4f" % (recall))
    if not os.path.exists("/opt/ml/output"):
        os.mkdir("/opt/ml/output")
    torch.save(model.state_dict(), f"/opt/ml/output/classifier_train.bin")
    print("")
    print("done")
