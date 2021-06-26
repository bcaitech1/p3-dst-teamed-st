# 💬 다중 도메인 대화 상태 추적

## 🏆최종 성적

- `Public LB`: Joint Goal Accuracy 0.8344 | 1등🥇
- `Private LB` : Joint Goal Accuracy 0.7335 | 1등🥇



## 📚Task Description

미리 정의된 시나리오의 대화에서 (System발화, User발화)를 하나의 턴으로 둘 때, 턴마다 순차적으로 유저 발화의 **Dialogue state(대화 상태)** 를 추적하는 Task

- ***기간*** : 2021.04.26 ~ 2021.05.21(4주)

- ***Dialogue State Tracking description*** :

	- `Input` : Dialogue 내에서 User와 System 발화 쌍 (1 Turn 단위)

	- `Output` : 해당 turn까지 누적된 Domain-Slot-Value의 pair

		![image](https://user-images.githubusercontent.com/38639633/122345725-23030d00-cf83-11eb-8023-e31719205950.png)

- ***Dataset Overview :*** Wizard-of-Seoul

	- 데이터는 아래와 같은 형식으로 구성되어 있으며, 예측해야하는 State는 **"Domain - Slot - Value"** 의 pair로 구성되어 있습니다. 

		- `Domain`: 5개 Class
		- `Slot` : 45개 Class

		![image](https://user-images.githubusercontent.com/38639633/122349426-37490900-cf87-11eb-9573-59351903c8bb.png)

- ***Metric*** : 모델은 **Joint Goal Accuracy**와 **Slot Accuracy**, 그리고 **Slot F1 Score** 세 가지로 평가됩니다.

	- **Joint Goal Accuracy**는 추론된 Dialogue State와 실제 Dialogue State의 **set**이 완벽히 일치하는지를 측정합니다. 즉, 여러 개의 Slot 중 하나라도 틀리면 0점을 받는 매우 혹독한 Metric입니다. 이에 반해, Slot Accuracy는 턴 레벨의 측정이 아닌 그 원소인 **(Slot, Value) pair**에 대한 Accuracy를 측정합니다. 심지어 아무런 Value를 추론하지 않고도 (== "none"으로 예측), 절반 이상의 점수를 낼 수 있는 매우 관대한 Metric입니다.

	- 따라서 본 대회에서는 JGA 다음으로 Slot-level의 F1 Score를 함께 평가합니다. ("none"의 영향력 약화)

	- 리더보드는 Joint Goal Accuracy → Slot F1 Score → Slot Accuracy로 소팅됩니다.

		![image](https://user-images.githubusercontent.com/38639633/123509101-9527d000-d6ae-11eb-83ff-574cf1248675.png)

<br/>

## :computer:Team Strategy 

[comment]: <> "아래 이미지는 주석"
[comment]: <> "![image]&#40;https://user-images.githubusercontent.com/38639633/119125512-d0f6c680-ba6c-11eb-952e-fdc6de36fef9.png&#41;"

![image](https://user-images.githubusercontent.com/48181287/119263872-c9c1eb00-bc1b-11eb-916c-f6e171f1ba79.png)



<br><br>

## 📁프로젝트 구조

```
p3-dst-teamed-st>
├── README.md
├── coco
│   ├── classifier_train.py
│   ├── data_utils.py
│   ├── evaluation.py
│   ├── gen_train.py
│   ├── model.py
│   ├── preprocessor.py
│   ├── pretrain.py
│   └── start_coco.py
├── data
│   ├── ontology.json
│   ├── slot_meta.json
│   ├── wos-v1_dev.json
│   └── wos-v1_train.json
├── data_utils.py
├── eval_utils.py
├── evaluation.py
├── hardvote_v2.py
├── inference.py
├── loss.py
├── model
│   ├── somdst.py
│   ├── sumbt.py
│   └── trade.py
├── preprocessor.py
├── requirements.txt
├── somdst_train.py
├── sumbt_train.py
└── trade_train.py
```

### :open_file_folder:File overview 

- `coco` : CoCo augmentation을 위한 data generation folder
- `data` : [KLUE WOS](https://klue-benchmark.com/tasks/73/data/description) banchmark dataset (2021.06 기준)
- `model` : 학습에 사용한 3가지 모델 Class
- `data_utils.py` : data util
- `eval_utils.py` : evaluation을 위한 utils
- `evaluation.py` : evaluation print
- `hardvote_v2.py` : 앙상블을 위한 hardvoting file
- `inference.py` : model prediction을 위한 fils
- `loss.py` : masked_cross_entropy loss
- `preprocessor.py` : model별 preprocessor
- `somdst_train.py` : som-dst 학습
- `sumbt_train.py` : sumbt 학습
- `trade_train.py` : trade 학습



<br><br>

## :page_facing_up:Installation 

#### Dependencies

- torch==1.7.0+cu101
- transformers==3.5.1

<!-- - pytorch-pretrained-bert -->

```
pip install -r requirements.txt
```

<br><br>

## 🧬Final Model

###  Trade

- Open vocab 기반의 DST model로 Unseen value를 맞출 수 있습니다.

- 모든 Slot을 전부 예측해야 하기 때문에 속도가 느리다는 단점이 있지만 그 단점을 보완하기 위해 Parallel decoding이 사용되었습니다.

- Utterance Encoder의 성능개선을 위해 bidirection RNN Encoder를 BERT로 교체하였습니다.

![스크린샷 2021-06-18 10 15 51](https://user-images.githubusercontent.com/67869514/122490995-18e21c80-d01e-11eb-93e5-5f44cced27a6.png)


- 사용법
```
# trade_train.py
python trade_train.py --save_dir ./output
```

<br><br>

### SUMBT

- Ontology 기반의 DST model로 이름같이 value의 갯수가 많은 slot에 유리합니다.
- Unseen value를 맞추지 못한다는 단점이 있지만 대회에서 open vocab 기반 모델인 SOM-DST의 output을 새로운 Ontology로 사용하여 개선하였습니다.

![](https://i.imgur.com/kNcXCxB.png)

- 사용법
```
# sumbt_train.py
python sumbt_train.py --save_dir ./output
```

<br><br>

### SOM-DST

- Open vocab 기반의 DST model 이며 TRADE의 모든 slot을 generation하는건 비효율 적이라는 단점을 보완하기위해 등장한 모델
- Utterance를 보고 UPDATE가 필요한 경우에만 generation


![](https://i.imgur.com/d82ZWqz.png)



- 사용법
```
# somdst_train.py
python somdst_train.py --save_dir ./output
```

<br><br>

### data augumentation

#### CoCo

- 자주 사용 되는 slot의 조합(ex. 택시-목적지, 도착-시간)이 아닌경우 맞추지 못하는 Counter factual을 지적한 논문
- pretrained된 BartForConditionalGeneration를 사용하여 utterance를 generation
- pretrained된 classifier로 state를 추출하고 role based Slot value match filter로 필터링을 거쳐진 utterance를 augumentation data로 사용.
![](https://i.imgur.com/EHq2uO3.png)

- :exclamation: 절대경로 사용에 주의
```
# get generation model, classifier model
# coco/pretrain.py
python pretrain.py

# coco/start_coco.py
python start_coco.py
```

<br><br>

### Ensemble

#### hardvoting

![](https://i.imgur.com/soAswyD.png)

`SLOT_FIRST_AND_TOP_VALUE`: 대분류인 슬롯에 먼저 투표를 한 뒤에, 해당 슬롯 안에서 가장 많은 표를 받은 value값을 선택

```
# hardvote_v2.py
python hardvot_v2.py mode=save --csv_dir=./output --save_dir=./hardvoting_result
```

<br><br>

## Reference

### paper

- [SUMBT: Slot-Utterance Matching for Universal and Scalable Belief Tracking](https://www.aclweb.org/anthology/P19-1546/)
- [TRADE: Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems](https://www.aclweb.org/anthology/P19-1078/)
- [SOM-DST:Efficient Dialogue State Tracking by Selectively Overwriting Memory](https://arxiv.org/abs/1911.03906)
- [TAPT : Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)
- [CoCo : Controllable Counterfactuals for Evaluating Dialogue State Trackers](https://arxiv.org/abs/2010.12850)

### Github

- [SUMBT github](https://github.com/SKTBrain/SUMBT)
- [TRADE github](https://github.com/jasonwu0731/trade-dst)
- [SOM-DST github](https://github.com/clovaai/som-dst)

### Dataset

- [MultiWOZ 2.1](https://paperswithcode.com/dataset/multiwoz)
- [KLUE:WOS](https://klue-benchmark.com/tasks/73/data/description)



## :man_technologist: Contributors

[윤도연(ydy8989)](https://github.com/ydy8989) | [전재열(Jayten)](https://github.com/jayten-jeon) | [설재환(anawkward)](https://github.com/anawkward) | [민재원(ekzm8523)](https://github.com/ekzm8523) | [김봉진(BongjinKim)](https://github.com/BongjinKim) | [오세민(osmosm7)](https://github.com/osmosm7)







