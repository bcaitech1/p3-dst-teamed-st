import torch

from tqdm import tqdm

def inference(model, eval_loader, processor, device):
    model.eval()
    predictions = {}
    for batch in tqdm(eval_loader):
        input_ids, segment_ids, input_masks, target_ids, num_turns, guids = \
            [b.to(device) if not isinstance(b, list) else b for b in batch]

        with torch.no_grad():
            _, pred_slot = model(
                input_ids, segment_ids, input_masks, labels=None, n_gpu=1
            )

        batch_size = input_ids.size(0)
        for i in range(batch_size):
            guid = guids[i]
            states = processor.recover_state(pred_slot.tolist()[i], num_turns[i])
            for tid, state in enumerate(states):
                predictions[f"{guid}-{tid}"] = state
    return predictions