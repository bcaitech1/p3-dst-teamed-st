import numpy as np
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch


def metric(probs, labels, thresh):
    preds = (probs > thresh)
    preds = preds.cpu().numpy() * 1
    labels = labels.byte().cpu().numpy() * 1
    return preds, labels


def evaluate(model, device, eval_dataloader, thresh=0.5):

    model.eval()
    preds_list, label_list = [], []

    for step, batch in enumerate(tqdm(eval_dataloader)):
        input_ids, input_masks, segment_ids, target_ids = (b.to(device) for b in batch)

        input_ids = input_ids.to(device)
        input_mask = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        target_ids = target_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        probs = logits.sigmoid()
        preds, labels = metric(probs, torch.squeeze(target_ids), thresh)
        preds_list.append(preds)
        label_list.append(labels)

    all_pred = np.concatenate(preds_list, axis=0)
    all_label = np.concatenate(label_list, axis=0)
    precision = precision_score(all_label, all_pred, average="samples", zero_division=1)
    recall = recall_score(all_label, all_pred, average="samples", zero_division=1)

    return precision, recall