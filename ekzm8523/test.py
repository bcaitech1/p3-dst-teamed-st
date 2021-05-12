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
from inference import inference_trade, inference_sumbt
from model import TRADE, masked_cross_entropy_for_value, SUMBT
from preprocessor import TRADEPreprocessor, SUMBTPreprocessor
from torch.cuda.amp import autocast,  GradScaler
from pprint import pprint
import torch.cuda.amp as amp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    train_data_file = f"/opt/ml/input/data/train_dataset//train_dials.json"
    slot_meta = json.load(open(f"/opt/ml/input/data/train_dataset/slot_meta.json"))
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_data = train_data[:10]
    dev_data = dev_data[:10]
    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )
    tokenizer = BertTokenizer.from_pretrained("dsksd/bert-ko-small-minimal")
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[STATE]", "[EOS]"]})
    vocab_size = added_token_num + tokenizer.vocab_size
    print(tokenizer.all_special_tokens)
    processor = TRADEPreprocessor(slot_meta, tokenizer)
    train_feature = list(map(processor._convert_example_to_feature, train_examples))


    print(train_feature)
    for i in range(10):
        print(tokenizer.decode(train_feature[i].input_id))
    # for target_id in train_feature.target_ids:
    #     print(tokenizer.decode(target_id))

