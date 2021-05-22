import argparse
import os
import glob
import os
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer

from data_utils import WOSDataset, get_examples_from_dialogues, convert_state_dict
from models import SOMDST, masked_cross_entropy_for_value
from preprocessor import SOMDSTPreprocessor
import torch.cuda.amp as amp
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, default="../../ensemble")
    args = parser.parse_args()
    slot_meta = json.load(open("../../input/data/train_dataset/slot_meta.json",  'rt', encoding='UTF8'))
    # print(len(slot_meta))
    ensemble_lst = glob.glob(os.path.join(args.predictions_dir,'*.csv'))
    ensemble_dict = {}
    for idx, pred in enumerate(ensemble_lst):
        predfile = json.load(open(pred,  'rt', encoding='UTF8'))
        pred_df = pd.DataFrame(index=predfile.keys(), columns=slot_meta)
        pred_df
        # # print(predfile.keys())
        # for keys in predfile.keys():
        #     ensemble_dict[predfile[keys]]=
        adfasdf