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
from preprocessor import CoCoPreprocessor
import os
import pickle
from evaluation import evaluate
from model import BertForMultiLabelSequenceClassification
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score



def metric(probs, labels, thresh):
    preds = (probs > thresh)
    preds = preds.cpu().numpy() * 1
    labels = labels.byte().cpu().numpy() * 1
    return preds, labels


def evaluate(model, device, eval_batch_size, eval_data, thresh=0.5):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    model.eval()
    preds_list, label_list = [], []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        probs = logits.sigmoid()
        preds, labels = metric(probs, torch.squeeze(label_ids), thresh)
        preds_list.append(preds)
        label_list.append(labels)

    all_pred = np.concatenate(preds_list, axis=0)
    all_label = np.concatenate(label_list, axis=0)
    precision = precision_score(all_label, all_pred, average="samples", zero_division=1)
    recall = recall_score(all_label, all_pred, average="samples", zero_division=1)

    return precision, recall


if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_model_name = "hyunwoongko/kobart"
    gen_model = BartForConditionalGeneration.from_pretrained(gen_model_name)
    gen_tokenizer = PreTrainedTokenizerFast.from_pretrained(gen_model_name)



    cls_model_name = "dsksd/bert-ko-small-minimal"
    data = json.load(open('/opt/ml/input/data/train_dataset/train_dials.json'))
    slot_meta = json.load(open('/opt/ml/input/data/train_dataset/slot_meta.json'))
    cls_tokenizer = BertTokenizer.from_pretrained(cls_model_name)
    bert_config = BertConfig.from_pretrained(cls_model_name, num_labels=len(slot_meta))
    bert_config.model_name_or_path = cls_model_name
    bert_config.num_labels = len(slot_meta)
    cls_model = BertForMultiLabelSequenceClassification.from_pretrained(cls_model_name, config=bert_config)

    slot2idx = {slot: i for i, slot in enumerate(slot_meta)}
    idx2slot = {i: slot for i, slot in enumerate(slot_meta)}
    processor = CoCoPreprocessor(slot_meta, gen_tokenizer, cls_tokenizer, bert_config)

    # generation data
    if not os.path.exists("coco_data/coco_gen_examples.pkl"):
        gen_examples = []
        for dialogue in tqdm(data):
            gen_examples.extend(get_coco_examples_from_dialogue(dialogue))

        gen_features = []
        for example in tqdm(gen_examples):
            gen_features.append(processor.gen_convert_example_to_feature(example))

        with open("coco_data/coco_gen_examples.pkl", "wb") as f:
            pickle.dump(gen_examples, f)
        with open("coco_data/coco_gen_features.pkl", "wb") as f:
            pickle.dump(gen_features, f)
    else:
        with open("coco_data/coco_gen_examples.pkl", "rb") as f:
            gen_examples = pickle.load(f)
        with open("coco_data/coco_gen_features.pkl", "rb") as f:
            gen_features = pickle.load(f)

    gen_train_data = CoCoGenDataset(gen_features)
    gen_train_sampler = RandomSampler(gen_train_data)
    gen_train_loader = DataLoader(gen_train_data, batch_size=32, sampler=gen_train_sampler, collate_fn=processor.gen_collate_fn)


    ########################
    num_train_epochs = 10
    batch_size = 1
    lr = 5e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    ########################

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    t_total = len(gen_train_loader) * num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in gen_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in gen_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * warmup_ratio), num_training_steps=t_total
    )

    gen_model.to(device)
    print("")

    for epoch in range(num_train_epochs):
        gen_model.train()
        for step, batch in enumerate(gen_train_loader):
            input_ids, target_ids, input_masks = (b.to(device) for b in batch)
            decoder_input_ids = target_ids[:, :-1].contiguous()
            decoder_input_ids[decoder_input_ids == -100] = gen_tokenizer.pad_token_id
            labels = target_ids[:, 1:].clone()
            outputs = gen_model(input_ids,
                            attention_mask=input_masks,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels)
            loss = outputs[0]
            if n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"[{epoch}/{num_train_epochs}][{step}/{len(gen_train_loader)}] {loss}")

    torch.save(gen_model.state_dict(), "/opt/ml/model/gen_model.bin")

    """
    여기부터 classifier
    """


    if not os.path.exists("coco_data/coco_cls_examples.pkl"):
        cls_examples = []
        for dialogue in tqdm(data):
            cls_examples.extend(get_coco_examples_from_dialogue(dialogue))

        cls_features = []
        for example in tqdm(cls_examples):
            cls_features.append(processor.cls_convert_example_to_feature(example))

        with open("coco_data/coco_cls_examples.pkl", "wb") as f:
            pickle.dump(cls_examples, f)
        with open("coco_data/coco_cls_features.pkl", "wb") as f:
            pickle.dump(cls_features, f)

    else:
        with open("coco_data/coco_cls_examples.pkl", "rb") as f:
            cls_examples = pickle.load(f)
        with open("coco_data/coco_cls_features.pkl", "rb") as f:
            cls_features = pickle.load(f)

    cls_train_features, cls_dev_features = train_test_split(cls_features, test_size=0.2)

    cls_train_data = CoCoClassifierDataset(cls_train_features)
    cls_train_sampler = RandomSampler(cls_train_data)
    cls_train_loader = DataLoader(cls_train_data, batch_size=32, sampler=cls_train_sampler, collate_fn=processor.cls_collate_fn)

    cls_dev_data = CoCoClassifierDataset(cls_dev_features)
    cls_dev_sampler = SequentialSampler(cls_dev_data)
    cls_dev_dataloader = DataLoader(cls_dev_data, sampler=cls_dev_sampler, batch_size=32, collate_fn=processor.cls_collate_fn)

    ##########################
    epochs = 10
    batch_size = 32
    lr = 5e-5
    warmup_ratio = 0.1
    weight_decay = 0.01
    #########################

    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_total = len(cls_train_loader) * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cls_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in cls_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(t_total * warmup_ratio), num_training_steps=t_total
    )
    cls_model.to(device)

    print("Start training!!")
    for epoch in tqdm(range(epochs)):
        tr_loss = 0
        cls_model.train()
        for step, batch in enumerate(tqdm(cls_train_loader)):
            input_ids, input_masks, segment_ids, target_ids = (b.to(device) for b in batch)
            loss = cls_model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks, labels=target_ids)

            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(cls_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(f"[{epoch}/{epochs}][{step}/{len(cls_train_loader)}] loss : {loss}")
        precision, recall = evaluate(cls_model, device, cls_dev_dataloader)
        print("***** Eval on dev set *****")
        print("Current precision = %.4f" % (precision))
        print("Current recall = %.4f" % (recall))
    if not os.path.exists("/opt/ml/output"):
        os.mkdir("/opt/ml/output")
    torch.save(cls_model.state_dict(), f"/opt/ml/output/classifier_train.bin")
    print("")
    print("done")

