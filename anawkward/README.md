## Shared Documents

### Notice
<details><summary>Pull request</summary>

1. main 폴더에 update 혹은 create를 하실경우는 피어세션에서 라이브로 하거나 pull request를 통해 하기!
2. 수정할 때, 브랜치 확인 잘 하기 (main branch 말고 각자의 branch에서 작업후 add, commit, push branch, pr , merge 순서)
3. main folder에 반영하실때는 최소 한명이상의 코드리뷰

</details>

<details><summary>피어세션</summary>

9:00, 16:30

</details>

### Todos


<details><summary>Data</summary>

- private 셋과 public 셋, train 셋에 분포차이 있는지 확인하기.  
알려진 사실 : private 에는 의도적으로 unseen, counterfactual value가 추가돼있음
- 추가데이터 수집(WOZ dataset?)
- public lb pseudo labeling

</details>

<details><summary>Model</summary>

- Slot-value, turn encoder 
- SOTA : https://paperswithcode.com/sota/multi-domain-dialogue-state-tracking-on-1
- Woz Dataset 에서 SOTA모형들이 Wos Dataset에서도 높은 성능을 보일까? 영어 - 한국어의 분포차이, 데이터셋의 크기(1000 vs 9000)와 domain, label 차이 등
</details>

### Suggestion
- 깃허브의 코드를 바로 서버에서 실행하기  
- 6명 서버를 묶어서 사용하기