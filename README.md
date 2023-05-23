# SP_NeuBoots

**일반적인 linear regression에서의 성능을 1차적으로 확인하고 있음.**
--- 
일단 수정한 모델에 대한 디버깅은 끝나서 결과는 나온다. ~허나 metric에 대한 문제가 있는 것으로 보인다.~
- train에서 metric과 val 함수를 뜯어보기  
- loss가 굉장히 불안정하게 나온다. 이유를 찾아보기. 일단 loss가 전체적으로 떨어지는건 맞는지, 그리고 실제로 추론은 어느정도되는지 확인.
  
---  
- 지금 해야할 일 
  -  TEST 과정 metric 확인 
  - 일반적인 regression 을 해서 결과를 살펴보고 paper에서 제시된 방법론으로 했을 때와 차이를 살펴보자. 


- 더 확인해야할 부분
  - NbsCls in nbsnet.py : 마지막에 레이어를 추가하는 부분. 여기서 alpha가 int가 아닐 때 시행하는 부분이 이해안됨.
  - 시뮬레이션 데이터셋을 바꿀 때 이름을 바꿔서 추가하자. 아니면 계속 삭제해서 새로 돌려야 한다. 

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
 n_train= 10000
 n_test = 2000
 # true beta
 beta = torch.ones([400,1])
 # mean 5, std 2
 train_X = torch.normal(5,2,size=(n_train,400))
 train_y = torch.mm(train_X,beta)
 test_X = torch.normal(5,2,size=(n_test,400))
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
  batch_size: 20
  cpus: 1
  gpus: "0"
  seed: 0
  num_epoch: 200
  phase: train
  epoch_th: 0
  num_mc: 5 # ?
  n_a: 1
  adv_training: False

module:
  model:
    name: Reg_model
    num_classes: 1
    dropout_rate: 0.0

  optim:
    name: SGD
    lr: 0.05
    momentum: 0.9
    nesterov: True
    weight_decay: 0.001

  lr_scheduler:
    name: CosineAnnealingLR
    T_max: 200

  loss: [NbsLoss]
  loss_args:
    NbsLoss:
      reduction: mean
  loss_weight: [1.]
  val_metric: Accuracy
  test_metric: Accuracy
  metric_args:
    nlabels: 1
    reduction: mean 
```
