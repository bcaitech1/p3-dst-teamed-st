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

from model import BertForMultiLabelSequenceClassification
@dataclass
class CoCoClassifierInputExample:
    guid: str
    system_utter: str
    user_utter: str
    turn_state: List[str]


@dataclass
class CoCoClassifierInputFeature:
    input_id: List[int]
    target_id: List[int]
    token_type_id: List[int]

def get_coco_examples_from_dialogue(dialogue):
    """ Dialogue 데이터셋 파일 -> CoCoClassifierInputExamples """
    guid = dialogue["dialogue_idx"]
    examples = []
    d_idx = 0
    previous_state = []
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"]
        else:
            sys_utter = ""

        user_utter = turn["text"]
        state = turn.get("state")

        turn_state = sorted(list(set(state) - set(previous_state)))
        examples.append(CoCoClassifierInputExample(guid=f"{guid}-{d_idx}",
                                                   system_utter=sys_utter,
                                                   turn_state=turn_state,
                                                   user_utter=user_utter))

        d_idx += 1
        previous_state = state

    return examples


def convert_example_to_feature(example, tokenizer):
    """ CoCoClassifierInputExamples -> CoCoClassifierInputFeature """
    sys = tokenizer.tokenize(example.system_utter)
    user = tokenizer.tokenize(example.user_utter)
    sys_token = [tokenizer.bos_token] + sys + [tokenizer.eos_token]
    user_token = user + [tokenizer.eos_token]

    sys_id = tokenizer.convert_tokens_to_ids(sys_token)
    user_id = tokenizer.convert_tokens_to_ids(user_token)

    token_type_id = [0] * len(sys_token) + [1] * len(user_token)

    turn_state = [tokenizer.bos_token]
    for state in example.turn_state:
        turn_state += tokenizer.tokenize(state) + [tokenizer.eos_token]

    input_id = sys_id + user_id
    target_id = tokenizer.convert_tokens_to_ids(turn_state)

    return CoCoClassifierInputFeature(input_id=input_id, target_id=target_id, token_type_id = token_type_id)


def pad_ids(arrays, pad_idx, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
    return arrays


def collate_fn(batch):
    input_ids = torch.LongTensor(pad_ids([b.input_id for b in batch], tokenizer.pad_token_id))
    target_ids = torch.LongTensor(pad_ids([b.target_id for b in batch], -100))
    input_masks = input_ids.ne(tokenizer.pad_token_id).float()
    return input_ids, target_ids, input_masks


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/opt/ml/model/coco.bin"
    data = json.load(open('/opt/ml/input/data/train_dataset/train_dials.json'))

    bert_config = BertConfig.from_pretrained("bert-base-uncased", num_labels=len(processor.get_labels()))
    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-uncased", config=bert_config)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    slot_meta = json.load(open('/opt/ml/input/data/train_dataset/slot_meta.json'))

    examples = []
    for dialogue in tqdm(data):
        examples.extend(get_coco_examples_from_dialogue(dialogue))

    features = []
    for example in tqdm(examples):
        features.append(convert_example_to_feature(example, tokenizer))

