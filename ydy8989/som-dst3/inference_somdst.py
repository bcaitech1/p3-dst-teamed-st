import argparse
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
        _, op_ids = state_scores.view(-1, 6).max(-1)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/opt/ml/input/data/eval_dataset"
    )
    parser.add_argument("--model_dir", type=str, default="/opt/ml/result")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/predictions")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="SOMDST10/model-17.bin")

    args = parser.parse_args()
    # args.data_dir = os.environ["SM_CHANNEL_EVAL"]
    # args.model_dir = os.environ['SM_CHANNEL_MODEL']
    # args.output_dir = os.environ["SM_OUTPUT_DATA_DIR"]
    args.model_dir = os.path.join(args.model_dir, args.model_name)
    args.output_dir = os.path.join(args.output_dir, args.model_name.split("/")[0])
    model_dir_path = os.path.dirname(args.model_dir)
    eval_data = json.load(open(f"{args.data_dir}/eval_dials.json", "r"))
    config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
    config = argparse.Namespace(**config)
    slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))

    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    added_token_num = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[SLOT]", "[NULL]", "[EOS]"]}
    )
    # Define Preprocessor
    processor = SOMDSTPreprocessor(slot_meta, tokenizer, max_seq_length=512)
    eval_examples = get_examples_from_dialogues(
        eval_data, user_first=False, dialogue_level=False
    )

    # Extracting Featrues
    # eval_features = processor.convert_examples_to_features(eval_examples)
    # eval_data = WOSDataset(eval_features)
    # eval_sampler = SequentialSampler(eval_data)
    # eval_loader = DataLoader(
    #     eval_data,
    #     batch_size=args.eval_batch_size,
    #     sampler=eval_sampler,
    #     collate_fn=processor.collate_fn,
    # )
    print("# eval:", len(eval_data))

    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    model = SOMDST(config, 5, 6, processor.op2id["update"])

    ckpt = torch.load(args.model_dir, map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    predictions = inference(model, eval_examples, processor, device)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    json.dump(
        predictions,
        open(f"{args.output_dir}/predictions.csv", "w"),
        indent=2,
        ensure_ascii=False,
    )
