import json
from collections import Counter, defaultdict
from tqdm import tqdm
import os
from dataclasses import dataclass
from time import sleep

############################################## region 재료 ########################################################

# 디자인
'''
csv_file1, csv_file2
-> 딕셔너리{턴_id : 예측결과}
-> 하드보팅결과 = hardvoting(예측결과들)
-> 딕셔너리{턴_id : 하드보팅결과}
-> csv_file_voted
'''


@dataclass
class CRITERION:
    SV_MAJORITY1 = "sv_majority1"
    SV_MAJORITY2 = "sv_majority2"
    SLOT_FIRST_AND_TOP_VALUE = "slot_first_and_top_value"

    default = SV_MAJORITY2


@dataclass
class VERBOSE:
    # [progress bar, discord, discord_contents]
    FULL = (1, 1, 1)
    PROGRESS = (1, 1, 0)
    CONTENTS = (0, 0, 1)

    default = FULL


def sum_predictions(predictions: list, turn: str):
    slot_votes = []
    slotvalue_votes = []
    for p in predictions:
        s_v = svs2s_v(p[turn])
        slot_votes += [item[0] for item in s_v]
        slotvalue_votes += {item for item in s_v}
    slot_counts = Counter(slot_votes)
    slotvalue_counts = Counter(slotvalue_votes)

    return {"slot_counts": slot_counts, "slotvalue_counts": slotvalue_counts}


def hardvoting(sum_preds: dict, n_voter, criterion=CRITERION.default):
    criterion = criterion.lower()

    sum_slots = sum_preds['slot_counts']
    sum_sv = sum_preds['slotvalue_counts']

    if criterion == "sv_majority1":
        # slot-value 쌍을 합쳐서 하나의 값으로 보고 투표.

        # n_voter 가 1명일 때, 1표 이상 받으면 채택
        # n_voter 가 2명일 때, 1표 이상 받으면 채택
        # n_voter 가 3명일 때, 1표 이상 받으면 채택 <- 너무 관대함. 과다한 value를 생성하게 되는 문제.
        # n_voter 가 4명일 때, 2표 이상 받으면 채택
        # n_voter 가 5명일 때, 2표 이상 받으면 채택

        return {k: v for k, v in sum_sv.items() if v >= n_voter // 2}

    if criterion == "sv_majority2":
        # slot-value 쌍을 합쳐서 하나의 값으로 보고 투표.

        # n_voter 가 1명일 때, 1표 이상 받으면 채택
        # n_voter 가 2명일 때, 2표 이상 받으면 채택
        # n_voter 가 3명일 때, 2표 이상 받으면 채택 <- 1:1:1 일 때, 아무것도 선택을 안하는 엄격한 투표방식.
        # n_voter 가 4명일 때, 3표 이상 받으면 채택
        # n_voter 가 5명일 때, 3표 이상 받으면 채택
        if n_voter == 1:
            thres = 1
        else:
            thres = n_voter // 2 + 1

        return {k: v for k, v in sum_sv.items() if v >= thres}

    elif criterion == "slot_first_and_top_value":
        # slot 으로 먼저 투표. 과반수인 슬롯에 대해서 밸류를 항상 뽑기.
        if n_voter == 1:
            thres = 1
        else:
            thres = n_voter // 2 + 1

        won_slots = {k: v for k, v in sum_slots.items() if v >= thres}.keys()
        top_svs = dict()
        for s in won_slots:
            cs = Counter()
            for sv, c in sum_sv.items():
                if s == sv[0]:
                    cs += {sv[1]: c}
            top_value = cs.most_common(1)[0]
            top_svs[(s, top_value[0])] = top_value[1]
        return top_svs


def voting2preds(voting_result: dict):
    return {k: [f"{s}-{v}" for s, v in v.keys()] for k, v in voting_result.items()}


def save_csv(predictions, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fpath = os.path.join(save_dir, 'hardvoting_result.csv')
    if os.path.exists(fpath):
        print(f"{fpath} already exists. and is now overwritten")
    json.dump(predictions, open(fpath, "w"), indent=2, ensure_ascii=False, )


def voter_meta(predictions, results):
    results_pred = voting2preds(results)
    n_voter = len(predictions)
    win_count = [0 for _ in range(n_voter)]  # 고른게 당첨되면 count 상승
    vote_count = [0 for _ in range(n_voter)]
    n_turns = (len(results_pred))

    for turn, accepted in results_pred.items():
        for i in range(n_voter):
            voter = predictions[i]
            hits = sum([1 for item in voter[turn] if item in accepted])
            win_count[i] += hits
            vote_count[i] += len(voter[turn])

    win_rate = [f"{win_count[i] / vote_count[i] * 100}%" for i in range(n_voter)]

    return dict(win_count=win_count, vote_count=vote_count, n_turns=n_turns, win_rate=win_rate)


def sv2s_v(sv):
    s1, s2, v = sv.split('-')
    s = '-'.join([s1, s2])
    return s, v


def svs2s_v(state):
    retList = []
    for sv in state:
        s, v = sv2s_v(sv)
        retList.append((s, v))
    return retList


def show_democarcy(turns, predictions, n_voter, voted_predictions, verbose, delay):
    prev_guid = ""
    if verbose[0] == 1:
        gen = tqdm(turns)
    else:
        gen = turns
    for t in gen:
        if prev_guid == t.split('-')[-2]:
            continue
        sum_preds_at_t = sum_predictions(predictions, turn=t)
        for v in sum_preds_at_t['slotvalue_counts'].values():
            if v != n_voter:
                states = [p[t] for p in predictions]
                allSet = set()
                for s in states:
                    # print(s);sleep(1)
                    allSet.update(s)
                negatives = [set() for _ in range(len(states))]
                for i in range(len(states)):
                    s = states[i]
                    negatives[i] = allSet - set(s)
                uncommon = set()
                for n in negatives:
                    uncommon.update(n)
                common = allSet - uncommon
                if verbose[1] == 1:
                    print()
                    print(f"discord found at turn : {t} | common values : {len(common)}")
                    sleep(delay)
                for s in states:
                    uncommon_vote = set(s) - common
                    msgList = []
                    for u in uncommon_vote:
                        if u in voted_predictions[t]:
                            msgList.append(u + "(accepted)")
                        else:
                            msgList.append(u)
                    if verbose[2] == 1:
                        print(msgList)
                print()
                sleep(delay)
                prev_guid = t.split('-')[-2]
                break


# endregion

######################################## region 테스트 ########################################################
def debug():
    prediction_dir = "./"
    prediction_files = [file for file in os.listdir(prediction_dir) if file.endswith(".csv")]

    predictions = [json.load(open(p, "r")) for p in prediction_files]

    n_voter = len(predictions)

    assert n_voter >= 2, "not enough csv files to initialize debugging"

    turns = list(predictions[0].keys())

    for i in tqdm(range(len(turns))):
        turn = turns[i]
        if predictions[0][turn] == predictions[1][turn]:
            continue
        else:
            break

    example_turn = turn

    example_state0 = list(predictions[0][example_turn])
    example_state1 = list(predictions[1][example_turn])

    sum_predictions_at_t = sum_predictions(predictions, turn=example_turn)

    sum_slots_at_t = sum_predictions_at_t['slot_counts']
    sum_sv_at_t = sum_predictions_at_t['slotvalue_counts']

    hardvoting_for_turn = hardvoting(sum_preds=sum_predictions_at_t, n_voter=n_voter,
                                     criterion=CRITERION.SLOT_FIRST_AND_TOP_VALUE)
    hardvoting_for_turn = hardvoting(sum_preds=sum_predictions_at_t, n_voter=n_voter, criterion=CRITERION.SV_MAJORITY1)

    results = defaultdict(dict)

    results[example_turn] = hardvoting_for_turn


def do_test(csv_dir=".", criterion=CRITERION.SLOT_FIRST_AND_TOP_VALUE, verbose=VERBOSE.default, delay=1):
    cwd = os.getcwd()
    print(f"currently working in {cwd}")

    prediction_files = [file for file in os.listdir(csv_dir) if file.endswith(".csv")]
    print(f"{len(prediction_files)} csv files : {prediction_files} are found at {cwd}")

    predictions = [json.load(open(p, "r")) for p in prediction_files]

    n_voter = len(predictions)

    turns = list(predictions[0].keys())

    results = defaultdict(dict)

    for t in tqdm(turns):
        sum_preds_at_t = sum_predictions(predictions, turn=t)
        results[t] = hardvoting(sum_preds_at_t, n_voter, criterion=criterion)

    voter_statistics = voter_meta(predictions, results)
    voted_predictions = voting2preds(results)

    show_democarcy(turns, predictions, n_voter, voted_predictions, verbose, delay)

    # save_csv(voted_predictions, save_dir)
    print(f"voted_predictions are not saved in testing")

    print(f"voting result summary : {voter_statistics}")


# endregion

##################################### region 프로그램 ###################################################

def csvs_to_hardvoted_csv(csv_dir, criterion=CRITERION.SLOT_FIRST_AND_TOP_VALUE, save_dir="./hardvotin_result"):
    cwd = os.getcwd()
    print(f"currently working in {cwd}")

    prediction_files = [file for file in os.listdir(csv_dir) if file.endswith(".csv")]
    print(f"{len(prediction_files)} csv files : {prediction_files} are found at {cwd}")

    predictions = [json.load(open(p, "r")) for p in prediction_files]

    n_voter = len(predictions)

    turns = list(predictions[0].keys())

    results = defaultdict(dict)

    for t in tqdm(turns):
        sum_preds_at_t = sum_predictions(predictions, turn=t)
        results[t] = hardvoting(sum_preds_at_t, n_voter, criterion=criterion)

    voter_statistics = voter_meta(predictions, results)
    voted_predictions = voting2preds(results)

    save_csv(voted_predictions, save_dir)
    print(f"voted_predictions are saved at {save_dir}")

    print(f"voting result summary : {voter_statistics}")


# endregion

if __name__ == "__main__":
    # do_test(csv_dir = ".", criterion = CRITERION.SLOT_FIRST_AND_TOP_VALUE, verbose = VERBOSE.FULL, delay = 1)
    csvs_to_hardvoted_csv(csv_dir=".", criterion=CRITERION.SLOT_FIRST_AND_TOP_VALUE, save_dir="./hardvoting_result")
