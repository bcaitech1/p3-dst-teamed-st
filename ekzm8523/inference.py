import argparse
import os
import json

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer

from data_utils import (WOSDataset, get_examples_from_dialogues, tokenize_ontology)
from model import TRADE, SUMBT
from preprocessor import TRADEPreprocessor, SUMBTPreprocessor
from torch.cuda.amp import autocast,  GradScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def inference_sumbt(model, eval_loader, processor, device, use_amp):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids = \
            [b.to(device) if not isinstance(b, list) else b for b in batch]

        if use_amp:
            with torch.no_grad():
                with autocast(enabled=use_amp):
                    output, pred_slot = model(input_ids, segment_ids, input_masks, labels=None, n_gpu=1)
        else:
            with torch.no_grad():
                output, pred_slot = model(input_ids, segment_ids, input_masks, labels=None, n_gpu=1)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            guid = guids[i]
            states = processor.recover_state(pred_slot.tolist()[i], num_turns[i])
            for tid, state in enumerate(states):
                predictions[f"{guid}-{tid}"] = state
    return predictions

def inference_trade(model, eval_loader, processor, device):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in batch
        ]

        with torch.no_grad():
            o, g = model(input_ids, segment_ids, input_masks, 9)

            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/eval_dataset")
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/output")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--model", type=str, default="sumbt", help="select trade or sumbt")
    args = parser.parse_args()
    
    print(args)
    
    model_dir_path = os.path.join(args.model_dir, args.model)
    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))
    config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
    config = argparse.Namespace(**config)
    slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))
    
    config.model = args.model # 나중에 지울 것
    
    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    
    print(config)
    if args.model == "trade":
        processor = TRADEPreprocessor(slot_meta, tokenizer)
        eval_examples = get_examples_from_dialogues(
            eval_data, user_first=False, dialogue_level=False
        )
    elif args.model == "sumbt":
        ontology = json.load(open("/opt/ml/input/data/train_dataset/ontology.json"))
        max_turn = max([len(e['dialogue']) for e in eval_data])
        processor = SUMBTPreprocessor(slot_meta,
                                      tokenizer,
                                      ontology=ontology,
                                      max_seq_length=64,
                                      max_turn_length=max_turn)
        eval_examples = get_examples_from_dialogues(
            eval_data, user_first=True, dialogue_level=True
        )
        slot_type_ids, slot_values_ids = tokenize_ontology(ontology, tokenizer, 12)

    # Extracting Featrues
    eval_features = processor.convert_examples_to_features(eval_examples)
    eval_data = WOSDataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(
        eval_data,
        batch_size=args.eval_batch_size,
        sampler=eval_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# eval:", len(eval_data))
    # model 선언
    model_path = os.path.join(model_dir_path, "best_model.bin")
    if args.model == "trade":
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )
    
        model = TRADE(config, tokenized_slot_meta)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.to(device)
        print("Model is loaded")
        predictions = inference_trade(model, eval_loader, processor, device)
        
    elif args.model == "sumbt":
        num_labels = [len(s) for s in slot_type_ids]
        model = SUMBT(config, num_labels, device)
        model.initialize_slot_value_lookup(slot_values_ids, slot_type_ids)
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.to(device)
        print("Model is loaded")
        predictions = inference_sumbt(model, eval_loader, processor, device)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    json.dump(
        predictions,
        open(f"{args.output_dir}/{args.model}_model.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
