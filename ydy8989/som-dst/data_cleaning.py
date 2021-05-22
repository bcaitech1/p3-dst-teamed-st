# !pip install transformers

from difflib import SequenceMatcher
import json
from tqdm.notebook import tqdm
import nltk
from nltk.translate import bleu
from nltk.translate.bleu_score import SmoothingFunction
from transformers import AutoTokenizer

smoothie = SmoothingFunction().method4

tokenizer = AutoTokenizer.from_pretrained("dsksd/bert-ko-small-minimal")


def similar(gold, pred, only_bleu=False, with_bleu=False):
    # if len(a) > len(b) :
    #   a, b = b, a
    sim = SequenceMatcher(None, pred, gold).ratio()
    if not only_bleu and not with_bleu:
        return sim
    bleu_score = (bleu(tokenizer.tokenize(pred)[0], tokenizer.tokenize(gold)[0]) + bleu(pred, gold) + bleu(gold,
                                                                                                           pred)) / 3
    if only_bleu:
        return bleu_score
    return (sim + bleu_score) / 2


with open("output (12).csv", "rb") as f:
    a = json.load(f)
cnt = 0
all_cnt = 0
for key, values in a.items():
    for i, value in enumerate(values):
        if "시간" in value:
            continue
        if ":" in value or "=" in value or "&" in value or "(" in value or ")" in value:
            print(f" before : {value}", end="\t")

        if "=" in value:
            a[key][i] = value.replace(" = ", "=")

        if "&" in value:
            a[key][i] = value.replace(" & ", "&")
        if "(" in value:
            value = value.replace("( ", "(")
            value = value.replace(" ) ", ")")
            value = value.replace(" )", ")")
            if value[-1] == "역":
                value = value.replace(" (", "(")
            a[key][i] = value
        if "동대문" in value and value[-1] == "(":
            value = "관광-이름-동대문 (흥인지문)"
            a[key][i] = value
        if "월드컵 경기장(" in value:
            a[key][i] = "관광-이름-서울 월드컵 경기장 (상암 월드컵 경기장)"

        if ":" in value or "=" in value or "&" in value or "(" in value or ")" in value:
            print(f" after : {a[key][i]}")
            cnt += 1

        all_cnt += 1
print(f"{cnt} / {all_cnt}")

json.dump(
    a,
    open(f"post-cleaning.csv", "w"),
    indent=2,
    ensure_ascii=False,
)

with open("post-cleaning.csv", "rb") as f:
    predctions = json.load(f)
with open("total_ontology.json", "rb") as f:
    ontology = json.load(f)

all_values = sorted(set(sum(ontology.values(), [])))
name_values = sorted(
    set(ontology['관광-이름'] + ontology['식당-이름'] + ontology['숙소-이름'] + ontology['택시-도착지'] + ontology['택시-출발지']))
a = ontology['관광-이름'].copy()
b = ontology['숙소-이름'].copy()
c = ontology['식당-이름'].copy()
ontology['관광-이름'] = [n for n in name_values if n not in b + c + ontology['지하철-도착지'] + ontology['지하철-출발지']]
ontology['숙소-이름'] = [n for n in name_values if
                     n not in a + c + ontology['지하철-도착지'] + ontology['지하철-출발지'] + ontology['숙소-종류']]
ontology['식당-이름'] = [n for n in name_values if n not in a + b + ontology['지하철-도착지'] + ontology[
    '지하철-출발지'] and "하우스" not in n and "모텔" not in n and "호텔" not in n]

cnt = 0
for guid, states in tqdm(predctions.items()):
    flag = False
    for i, state in enumerate(states):
        dom, slot, value = state.split('-')
        domslot = f"{dom}-{slot}"
        if "시간" in slot:
            continue
        if value in all_values:
            continue
        # if ('게스트' in value  or '하우스' in value or '에어비'): #or '호텔' in value or '모텔' in value):# and '그린' not in value and '힐링' not in value and '송이' not in value:
        if '투데이' in value and ('게스트' in value or '하우스' in value or '에어비' in value or '호텔' in value or '모텔' in value):
            if len(value.split(' ')) > 1:
                value = ' '.join(value.split(' ')[:-1])
        if domslot in ['택시-출발지', '택시-도착지']:
            values = name_values
        else:
            values = ontology[domslot]
        values = [v for v in values if v != "none"]
        similarities = sorted([(v, similar(value, v, only_bleu=True)) for v in values], reverse=True,
                              key=lambda x: x[1])
        print(domslot, value, "\t=>\t", similarities[:3])

        predctions[guid][i] = f"{domslot}-{similarities[0][0]}"
        flag = True
    if flag:
        cnt += 1
print(f"{cnt} / 14771")
json.dump(
    predctions,
    open(f"post-cleaning.csv", "w"),
    indent=2,
    ensure_ascii=False,
)