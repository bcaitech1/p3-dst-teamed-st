# 💬 다중 도메인 대화 상태 추적

## Dialogue State Tracking

미리 정의된 시나리오의 대화에서 (System발화, User발화)를 하나의 턴으로 둘 때, 턴마다 순차적으로 유저 발화의 **Dialogue state(대화 상태)** 를 추적하는 Task

![image](https://user-images.githubusercontent.com/38639633/122345725-23030d00-cf83-11eb-8023-e31719205950.png)

> `Input` : ["안녕하세요.", "네. 안녕하세요. 무엇을 도와드릴까요?", "서울 중앙에 위치한 호텔을 찾고 있습니다. 외국인 친구도 함께 갈 예정이라서 원활하게 인터넷을 사용할 수 있는 곳이 었으면 좋겠어요."]
>
> `Output` : ["숙소-지역-서울 중앙", "숙소-인터넷 가능-yes"]

<br><br>

## Dataset

![image](https://user-images.githubusercontent.com/38639633/122349426-37490900-cf87-11eb-9573-59351903c8bb.png)

- 데이터는 위와 같은 형식으로 구성되어 있으며, 예측해야하는 State는 **"Domain - Slot - Value"**의 pair로 구성되어 있습니다. 

	- Domain : 5개 Class
	- Slot : 45개 Class

	

<br><br>

## Team Score

**Public** : Joint Goal Accuracy 0.8344, 1등🥇

**Private** : Joint Goal Accuracy 0.7335, 1등🥇

[comment]: <> "아래 이미지는 주석"
[comment]: <> "![image]&#40;https://user-images.githubusercontent.com/38639633/119125512-d0f6c680-ba6c-11eb-952e-fdc6de36fef9.png&#41;"
![image](https://user-images.githubusercontent.com/48181287/119263872-c9c1eb00-bc1b-11eb-916c-f6e171f1ba79.png)

### 네트워킹 데이 발표 링크

[https://drive.google.com/file/d/1Ws2XHhEHmObsl64gk49roNXXsDjZxGBv/view](https://drive.google.com/file/d/1Ws2XHhEHmObsl64gk49roNXXsDjZxGBv/view)



<br><br>

## Installation

#### Dependencies

- torch==1.7.0+cu101
- transformers==3.5.1


<!-- - pytorch-pretrained-bert -->

```
pip install -r requirements.txt
```

<br><br>

## 사용한 모델

####  Trade

- Open vocab 기반의 DST model로 Unseen value를 맞출 수 있습니다.

- 모든 Slot을 전부 예측해야 하기 때문에 속도가 느리다는 단점이 있지만 그 단점을 보완하기 위해 Parallel decoding이 사용되었습니다.

- Utterance Encoder의 성능개선을 위해 bidirection RNN Encoder를 BERT로 교체하였습니다.

![](https://i.imgur.com/d82ZWqz.png)

- 사용법
```
# trade_train.py
python trade_train.py --save_dir ./output
```

<br><br>

#### SUMBT

- Ontology 기반의 DST model로 이름같이 value의 갯수가 많은 slot에 유리합니다.
- Unseen value를 맞추지 못한다는 단점이 있지만 대회에서 open vocab 기반 모델인 SOM-DST의 output을 새로운 Ontology로 사용하여 개선하였습니다.

![](https://i.imgur.com/kNcXCxB.png)

- 사용법
```
# sumbt_train.py
python sumbt_train.py --save_dir ./output
```

<br><br>

#### SOM-DST

- Open vocab 기반의 DST model 이며 TRADE의 모든 slot을 generation하는건 비효율 적이라는 단점을 보완하기위해 등장한 모델입니다.
- Utterance를 보고 UPDATE가 필요한 경우에만 generation을 합니다.


![](https://i.imgur.com/d82ZWqz.png)



- 사용법
```
# somdst_train.py
python somdst_train.py --save_dir ./output
```

<br><br>

## data augumentation

#### CoCo

- 자주 사용 되는 slot의 조합(ex. 택시-목적지, 도착-시간)이 아닌경우 맞추지 못하는 Counter factual을 지적한 논문입니다.
- pretrained된 BartForConditionalGeneration를 사용하여 utterance를 generation합니다.
- pretrained된 classifier로 state를 추출하고 role based Slot value match filter로 필터링을 거쳐진 utterance를 augumentation data로 사용합니다.
![](https://i.imgur.com/EHq2uO3.png)

- 사용법 / 절대경로를 잘 지정해야 합니다.
```
# get generation model, classifier model
# coco/pretrain.py
python pretrain.py

# coco/start_coco.py
python start_coco.py
```

<br><br>

## Ensemble

#### hardvoting

![](https://i.imgur.com/soAswyD.png)

SLOT_FIRST_AND_TOP_VALUE : 대분류인 슬롯에 먼저 투표를 한 뒤에, 해당 슬롯 안에서 가장 많은 표를 받은 value값을 선택하는 알고리즘

```
# hardvote_v2.py
python hardvot_v2.py mode=save --csv_dir=./output --save_dir=./hardvoting_result
```


<br><br>

## Contributors

[윤도연(ydy8989)](https://github.com/ydy8989) | [전재열(Jayten)](https://github.com/Jayten) | [설재환(anawkward)](https://github.com/anawkward) | [민재원(ekzm8523)](https://github.com/ekzm8523) | [김봉진(BongjinKim)](https://github.com/BongjinKim) | [오세민(osmosm7)](https://github.com/osmosm7)







