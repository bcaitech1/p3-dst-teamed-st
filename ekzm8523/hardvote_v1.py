import json
from collections import Counter, defaultdict
import tqdm
import os

# 디자인
'''
csv_file1, csv_file2
-> 딕셔너리{턴_id : 예측결과}
-> 하드보팅결과 = hardvoting(예측결과들)
-> 딕셔너리{턴_id : 하드보팅결과}
-> csv_file_voted
'''

############################################## 재료 ########################################################

def sum_predictions(predictions:list, turn:str):
  states = []
  for p in predictions:
    states += p[turn]
  counts = Counter(states)

  return counts

def hardvoting(counts:Counter, n_voter):
    # n_voter 가 2명일 때, 1표 이상 받으면 채택
    # n_voter 가 3명일 때, 1표 이상 받으면 채택
    # n_voter 가 4명일 때, 2표 이상 받으면 채택
    return {k : v for k, v in counts.items() if v >= n_voter//2}

def voting2preds(voting_result:dict):
  return {k: list(v.keys()) for k, v in voting_result.items()}

def save_csv(predictions, save_dir):
    os.makedirs(save_dir, exist_ok = True)
    fpath = os.path.join(save_dir, 'hardvoting_result.csv')
    if os.path.exists(fpath):
        print(f"{fpath} already exists. and is now overwritten")
    json.dump(predictions, open(fpath, "w"), indent=2, ensure_ascii=False, )

def voter_meta(predictions, results):
    n_voter = len(predictions)
    win_count = [0 for _ in range(n_voter)]  # 고른게 당첨되면 count 상승
    vote_count = [0 for _ in range(n_voter)]
    n_turns = (len(results.keys()))

    for turn, accepted in voting2preds(results).items():
        for i in range(n_voter):
            voter = predictions[i]
            hits = sum([1 for item in voter[turn] if item in accepted])
            win_count[i] += hits
            vote_count[i] += len(voter[turn])

    win_rate = [f"{round(win_count[i] / vote_count[i] * 100)}%" for i in range(n_voter)]

    return dict(win_count = win_count, vote_count = vote_count, n_turns = n_turns, win_rate1 = win_rate)

######################################## 테스트 ########################################################
def debug():

    prediction_dir = "./"
    prediction_files = [file for file in os.listdir(prediction_dir) if file.endswith(".csv")]

    predictions = [json.load(open(p, "r")) for p in prediction_files]

    n_voter = len(predictions)

    assert n_voter >= 2, "not enough csv files to initialize debugging"

    turns = list(predictions[0].keys())

    example_turn = turns[0]
    example_state0 = list(predictions[0][example_turn])
    example_state1 = list(predictions[1][example_turn])

    sum_preds_at_t = sum_predictions(predictions, turn = example_turn)
    hardvoting_for_turn = hardvoting(sum_preds_at_t, n_voter)

    results = defaultdict(dict)

    results[example_turn] = hardvoting_for_turn

##################################### 프로그램 ###################################################

def csvs_to_hardvoted_csv(csv_dir, save_dir = "./hardvotin_result"):
    cwd = os.getcwd()
    print(f"currently working in {cwd}")

    prediction_files = [file for file in os.listdir(csv_dir) if file.endswith(".csv")]
    print(f"{len(prediction_files)} csv files : {prediction_files} are found at {cwd}")

    predictions = [json.load(open(p, "r")) for p in prediction_files]

    n_voter = len(predictions)

    turns = list(predictions[0].keys())

    results = defaultdict(dict)

    for t in tqdm.tqdm(turns):
        sum_preds_at_t = sum_predictions(predictions, turn = t)
        results[t] = hardvoting(sum_preds_at_t, n_voter)

    voter_statistics = voter_meta(predictions, results)
    voted_predictions = voting2preds(results)

    save_csv(voted_predictions, save_dir)
    print(f"voted_predictions are saved at {save_dir}")

    print(f"voting result summary : {voter_statistics}")

if __name__ == "__main__":
  csvs_to_hardvoted_csv(csv_dir = ".", save_dir = "./hardvoting_result")