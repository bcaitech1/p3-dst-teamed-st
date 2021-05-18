from data_utils import (
    get_coco_examples_from_dialogues,
    coco_generator,
    convert_example_to_feature,
    CoCoClassifierInputExample,
    CoCoClassifierDataset,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from evaluation import evaluate
import json
import os
import pickle
import torch
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from transformers import BartForConditionalGeneration, BartConfig
from transformers import PreTrainedTokenizerFast, BertConfig, BertTokenizer
from model import BertForMultiLabelSequenceClassification
from preprocessor import CoCoClassifierPreprocessor

device = torch.device('cuda')
data_dir = "../../input/data/train_dataset"
model_path = "hyunwoongko/kobart"
generator_ckpt = "./checkpoint/gen_model.bin"
classifier_ckpt = "./checkpoint/classifier_train.bin"

data = json.load(open(os.path.join(data_dir, "train_dials.json"), "rt", encoding='UTF8'))
coco_examples = get_coco_examples_from_dialogues(data, dialogue_level=True)

gen_config = BartConfig.from_pretrained(model_path)
generator = BartForConditionalGeneration.from_pretrained(model_path, config=gen_config)

# gen_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
# state_dict = torch.load(generator_ckpt)
# generator.load_state_dict(state_dict)
print(generator)