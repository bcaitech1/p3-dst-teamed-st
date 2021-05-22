"""
디버깅 및 테스트용 파일
"""

from inference_somdst import *
import pickle

def wrong_output(predictions, labels, model_path):

    wrong_dict = {}
    for pred_key, pred_values in predictions.items():
        label_values = labels[pred_key]
        pred_values = sorted(pred_values)
        label_values = sorted(label_values)
        for pred_value in pred_values:
            if not pred_value in label_values:
                wrong_dict[pred_key] = {"labels": label_values, "preds:": pred_values}
                break
    json.dump(
        wrong_dict,
        open(f"{model_path}_wrong.json", "w"),
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/opt/ml/output/somdst_not_MP/best_model0"

    model_dir_path = os.path.dirname(model_path)
    model_name = model_path.split('/')[-1]


    config = json.load(open(f"{model_dir_path}/exp_config.json", "r"))
    config = argparse.Namespace(**config)

    with open(os.path.join(config.data_dir, "dev_somdst_examples.pkl"), "rb") as f:
        dev_examples = pickle.load(f)
    with open(os.path.join(config.data_dir, "dev_somdst_labels.pkl"), "rb") as f:
        dev_labels = pickle.load(f)


    slot_meta = json.load(open(f"{model_dir_path}/slot_meta.json", "r"))
    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    added_token_num = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[SLOT]", "[NULL]", "[EOS]"]}
    )
    # Define Preprocessor
    processor = SOMDSTPreprocessor(slot_meta, tokenizer, max_seq_length=512)
    model = SOMDST(config, 5, 4, processor.op2id["update"])

    ckpt = torch.load(f"{model_path}.bin", map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    print("Model is loaded")

    predictions = inference(model, dev_examples, processor, device)

    wrong_output(predictions, dev_labels, model_path)

