import json
import pickle
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizer

from data_utils import WOSDataset
from inference import inference_trade
from model import TRADE
from preprocessor import TRADEPreprocessor


from dataclasses import dataclass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = True

if __name__ == "__main__":
    model_dir = "/opt/ml/output/trade3"
    model_args = json.load(open(f"{model_dir}/exp_config.json"))
    slot_meta = json.load(open(f"{model_args['data_dir']}/slot_meta.json"))  # 45개의

    @dataclass
    class args:
        data_dir = model_args["data_dir"]
        model_dir = model_args["model_dir"]
        model = model_args["model"]
        train_batch_size = model_args["train_batch_size"]
        eval_batch_size = model_args["train_batch_size"]
        learning_rate = model_args["learning_rate"]
        adam_epsilon = model_args["adam_epsilon"]
        max_grad_norm = model_args["max_grad_norm"]
        epochs = model_args["epochs"]
        warmup_ratio = model_args["warmup_ratio"]
        random_seed = model_args["random_seed"]
        model_name_or_path = model_args["model_name_or_path"]
        hidden_size = model_args["hidden_size"]
        vocab_size = model_args["vocab_size"]
        hidden_dropout_prob = model_args["hidden_dropout_prob"]
        proj_dim = model_args["proj_dim"]
        teacher_forcing_ratio = model_args["teacher_forcing_ratio"]
        wandb_name = model_args["wandb_name"]
        n_gate = model_args["n_gate"]

    with open('trade_data/train_features_test.bin', 'rb') as f:
        train_features = pickle.load(f)
    with open('trade_data/dev_features_test.bin', 'rb') as f:
        dev_features = pickle.load(f)
    with open('trade_data/dev_labels_test.bin', 'rb') as f:
        dev_labels = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    added_token_num = tokenizer.add_special_tokens({"additional_special_tokens": ["[STATE]"]})
    vocab_size = added_token_num + tokenizer.vocab_size
    processor = TRADEPreprocessor(slot_meta, tokenizer)

    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )

    model = TRADE(args, tokenized_slot_meta, slot_meta)
    model.to(device)
    model.load_state_dict(torch.load(model_dir + "/best_model1.bin"))

    train_data = WOSDataset(train_features)
    train_sampler = SequentialSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.eval_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,

    )
    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,

    )

    # train_predictions = inference_trade(model, train_loader, processor, device)
    dev_predictions = inference_trade(model, dev_loader, processor, device)

    # wrong_value, wrong_slot = eval_wrong_count(predictions, dev_labels)
    # eval_result = _evaluation(predictions, dev_labels, slot_meta)
    #
    # print(predictions)