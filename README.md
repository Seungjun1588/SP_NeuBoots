# SP_NeuBoots

~**일반적인 linear regression에서의 성능을 1차적으로 확인하고 있음.**~  
**이제 nonlinear regression 혹은 다른 regression에서 성능을 확인해보기**
--- 
  
---  
- 지금 해야할 일 
  -  TEST 과정 metric 확인 
  - 일반적인 regression 을 해서 결과를 살펴보고 paper에서 제시된 방법론으로 했을 때와 차이를 살펴보자. 
  - 결과는 문제없는데, log에 TEST loss가 여전히 이상하게 나온다. 함수 정의한 부분 한번 더 확인하기.

- 더 확인해야할 부분
  - 시뮬레이션 데이터셋을 바꿀 때 이름을 바꿔서 추가하자. 아니면 계속 삭제해서 새로 돌려야 한다. 
  - 아무리 봐도 validation 할 때는 dirichlet가 아니라 그냥 rand_like 함수를 사용하고 있는 것으로 보인다. 확인해보자. 
---
- 문제가 있을 수 있는 부분
  - 데이터 사이즈 혹은 n_a의 사이즈 문제 -> paper와 최대한 유사한 세팅으로 확인 가능
  - metric의 잘못된 정의 -> 뜯어보는 중 Accuracy를 이용하고 있음
    - Accuracy가 파일 내에서 정의된 함수였는데, 이를 다른 걸로 바꿔야할 듯 하다. 
    - 일단 Nbsloss가 존재하니까 이걸 먼저 뜯어보자.
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
