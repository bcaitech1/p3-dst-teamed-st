import argparse
import os
import json

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from data_utils import WOSDataset, get_examples_from_dialogues, convert_state_dict
from model import SOMDST
from preprocessor import SOMDSTPreprocessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_state(state):
    for i, s in enumerate(state):
        s = s.replace(" : ", ":")
        state[i] = s.replace(" , ", ", ")
    return state


def inference(model, eval_examples, processor, device):
    processor.reset_state()
    model.eval()
    predictions = {}
    last_states = {}
    for example in tqdm(eval_examples):
        if not example.context_turns:
            last_states = {}
        # example.prev_state = last_states
        features = processor._convert_example_to_feature(example)
        features = processor.collate_fn([features])
        batch = [
            b.to(device) if not isinstance(b, int) and not isinstance(b, list) else b
            for b in features
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
        domain_scores, state_scores, gen_scores = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            slot_positions=slot_position_ids,
            attention_mask=input_masks,
            max_value=9,
            op_ids=None,
        )
        _, op_ids = state_scores.view(-1, 4).max(-1)
        if gen_scores.size(1) > 0:
            generated = gen_scores.squeeze(0).max(-1)[1].tolist()
        else:
            generated = []

        pred_ops = [processor.id2op[op] for op in op_ids.tolist()]
        processor.prev_state = last_states
        prediction = processor.recover_state(pred_ops, generated)
        prediction = postprocess_state(prediction)
        last_states = convert_state_dict(prediction)
        predictions[guids[0]] = prediction
    return predictions


def direct_output(model_path=None, model=None, processor=None):
    """
    model,processor 혹은 model_path 둘중 하나는 정확히 넣어주어야 실행됩니다.
    모델과 processor를 제대로 맞추지 않으면 오류가 납니다.
    실행 방법 두가지
    1. 저장된 모델 inference -> model_path 넣어주기 (확장자 제외, bin으로 통일)
    2. 학습중에 바로 inference -> model과 processor 넣어주기
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "/opt/ml/input/data/eval_dataset"

    eval_data = json.load(open(f"{data_path}/eval_dials.json", "r"))
    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=False, dialogue_level=False
    )
    if not processor:
        model_dir_path = os.path.dirname(model_path)
        model_name = model_path.split('/')[-1]

        config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
        config = argparse.Namespace(**config)

        slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))
        tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
        added_token_num = tokenizer.add_special_tokens(     # config 파일에 이미 vocab size가 변경되어 있음
            {"additional_special_tokens": ["[SLOT]", "[NULL]", "[EOS]"]}
        )
        # Define Preprocessor
        processor = SOMDSTPreprocessor(slot_meta, tokenizer, max_seq_length=512)
        tokenized_slot_meta = []
        for slot in slot_meta:
            tokenized_slot_meta.append(
                tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
            )

        model = SOMDST(config, 5, 4, processor.op2id["update"])

        ckpt = torch.load(f"{model_path}.bin", map_location="cpu")
        model.load_state_dict(ckpt)
        model.to(device)
        print("Model is loaded")

    predictions = inference(model, eval_examples, processor, device)


    json.dump(
        predictions,
        open(f"{model_dir_path}/{model_name}.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
