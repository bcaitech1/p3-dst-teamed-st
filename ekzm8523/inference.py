import json
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from data_utils import (WOSDataset, get_examples_from_dialogues, tokenize_ontology, convert_state_dict)
from torch.cuda.amp import autocast,  GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True

def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def inference_sumbt(model, eval_loader, processor, device):
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
            if use_amp:
                with autocast(enabled=use_amp):
                    o, g = model(input_ids, segment_ids, input_masks, 9)
            else:
                o, g = model(input_ids, segment_ids, input_masks, 9)
            _, generated_ids = o.max(-1)
            _, gated_ids = g.max(-1)

        for guid, gate, gen in zip(guids, gated_ids.tolist(), generated_ids.tolist()):
            prediction = processor.recover_state(gate, gen)
            prediction = postprocess_state(prediction)
            predictions[guid] = prediction
    return predictions


def teade_test_inference(model, eval_examples, processor, device):
    model.eval()
    predictions = {}
    for example in tqdm(eval_examples):

        if not example.context_turns:
            processor.pre_labels = []

        features = processor._convert_example_to_feature(example)
        features = processor.collate_fn([features])

        input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
            b.to(device) if not isinstance(b, list) else b for b in features
        ]
        with torch.no_grad():
            o, g = model(input_ids, segment_ids, input_masks, 9)

        _, generated_ids = o.max(-1)
        _, gated_ids = g.max(-1)

        gen = generated_ids.squeeze().tolist()
        gate = gated_ids.squeeze().tolist()
        prediction = processor.recover_state(gate, gen)
        prediction = postprocess_state(prediction)
        processor.pre_labels = prediction
        predictions[guids[0]] = prediction

    return predictions


def save_trade(model, processor, device, save_dir, epoch=-1):
    eval_data = json.load(open(f"/opt/ml/input/data/eval_dataset/eval_dials.json", "r"))

    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=False, dialogue_level=False
    )

    # Extracting Featrues
    eval_features = processor.convert_examples_to_features(eval_examples)
    eval_data = WOSDataset(eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_loader = DataLoader(
        eval_data,
        batch_size=8,
        sampler=eval_sampler,
        collate_fn=processor.collate_fn,
    )

    predictions = inference_trade(model, eval_loader, processor, device)
    json.dump(predictions, open(f'{save_dir}/predictions{epoch}.csv', 'w'), indent=2, ensure_ascii=False)

