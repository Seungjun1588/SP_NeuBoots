# SP_NeuBoots

~**일반적인 linear regression에서의 성능을 1차적으로 확인하고 있음.**~  
**이제 nonlinear regression 혹은 다른 regression에서 성능을 확인해보기**
--- 
- **방향성**
  -   ~이론적인 bootstrap과의 차이점을 생각해보자.~
  -   **point reference data를 시뮬레이션 하고(GP), 이를 baysian krigging 했을 때, ML로 krigging 했을 떄, NeuBoots를 이용해서 했을 때를 실험적으로 비교하자.**
  -   NeuBoots를 이용할 때는 BANP를 보고 시뮬레이션을 어떻게 했는지 참조하는게 좋은 것 같다. -> 여기서는 std를 같이 추정하기 때문에 가능하다. 
  -   이후, SNGP라는게 뭔지도 공부해보자 .(https://www.tensorflow.org/tutorials/understanding/sngp?hl=ko)
  -   krigging 하는 문제로 접근해볼까? x를 랜덤으로 일부만 주는 형태로 한다던지..(지금은 x가 fixed 되어있음.)
  -   구름 등에 의해서 가려진 상황을 가정, 시뮬레이션 데이터셋을 만드는 것은 가능할 것으로 보인다. 
  -   조금 더 현실적으로 만들려면, context 데이터의 개수도 랜덤으로 받게 만들어야 하는데 이는 batch 단위의 연산이 안되는 문제가 있다. 일단 이는 패스하고 해보자. 
  -  gaussian 예제뿐 아니라, non-gaussian에도 적용해볼 수 있을 듯 하다 
---  
- 지금 해야할 일 
  -  ~TEST 과정 metric 확인~ 
  - 일반적인 regression 을 해서 결과를 살펴보고 paper에서 제시된 방법론으로 했을 때와 차이를 살펴보자. 
  - 일반적인 regression에서 y에 대한 분포의 parameter estimation을 시도해보고 결과를 보자. 
  - ~결과는 문제없는데, log에 TEST loss가 여전히 이상하게 나온다. 함수 정의한 부분 한번 더 확인하기.~
  - linear regression 예제를 만드는데 부족한 부분이 있었는데, 좋은 참고자료를 찾았다. 아래의 링크를 참고해서 보충해보자. 
    -   https://github.com/deepmind/neural-processes
  - attention base 모델로 구현할 때, 시공간 상관관계를 같이 고려해서 estimation이 가능하도록 만들 수 있지 않을까?

- 더 확인해야할 부분
  - 아무리 봐도 validation 할 때는 dirichlet가 아니라 그냥 rand_like 함수를 사용하고 있는 것으로 보인다. 확인해보자. 
  - custom2의 경우 fine tuning하면 결과가 잘 나올 가능성이 보인다. 나중에 조금 더 시도해보자. 
  - ~activation을 바꾸는 것도 성능에 큰 영향을 미칠 것으로 보인다.~ 
---
- 문제가 있을 수 있는 부분
  - 데이터 사이즈 혹은 n_a의 사이즈 문제 -> paper와 최대한 유사한 세팅으로 확인 가능
  - metric의 잘못된 정의 -> 뜯어보는 중 Accuracy를 이용하고 있음
  - world size 설정(이 부분은 분산컴퓨팅에 대한 이해를 하고 설정을 건드려봐야할 듯 함)
  - 마지막 레이어에 w가 곱해지고(이건 alpha), loss에도 w를 곱한다(이건 그냥 가중치인가?). 이게 맞는 흐름인지 다시 한번 확인할 필요있음
    - 마지막 레이어에 곱해지는 w는 전체 alpha
    - loss에 곱하는 w는 batch size크기의 alpha. 즉, 하나의 batch 안에서는 같은 값을 곱하고 있다. 
  - n_a가 하는 역할 기록해놓기(계속 까먹는다. ) -> alpha vector size ( not batch size!)
---


   
- 현재 시뮬레이션 데이터셋
```python
n_train= 100000
n_test = 2000
# true beta
beta = torch.ones([100,1])
# mean 5, std 2
train_X = torch.normal(0,1,size=(n_train,100))
train_y = torch.mm(train_X,beta)
test_X = torch.normal(0,1,size=(n_test,100))
test_y = torch.mm(test_X,beta)
```
- 시뮬레이션 세팅 바꾸는 법(까먹지 않게 기록)
  - data_loader.py에서 시뮬레이션 코드를 변경.
  - __init__.py에서 dict에서 해당 모델의 값을 변경.
  - models/regression.py에서 모델의 구조를 변경.

---
- yaml 파일 세팅

```python
path:
  dataset: custom
  postfix: "train"

setup:
  model_type: nbs
  batch_size: 100
  cpus: 2
  gpus: "0"
  seed: 0
  num_epoch: 20
  phase: train
  epoch_th: -1
  num_mc: 20 
  n_a: 100
  adv_training: False

module:
  model:
    name: Reg_model
    num_classes: 1
    dropout_rate: 0.

  optim:
    name: SGD
    lr: 0.0005
    momentum: 0.9
    nesterov: True
    weight_decay: 0.001

  lr_scheduler:
    name: CosineAnnealingLR
    T_max: 200

  loss: NbsLoss
  loss_args:
    reduction: mean 
  loss_weight: [1.]
  val_metric: NbsLoss
  test_metric: NbsLoss
  metric_args:
    reduction: mean 
```
