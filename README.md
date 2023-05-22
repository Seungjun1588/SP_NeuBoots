# SP_NeuBoots

**일반적인 linear regression에서의 성능을 1차적으로 확인하고 있음.**
--- 
일단 수정한 모델에 대한 디버깅은 끝나서 결과는 나온다. 허나 metric에 대한 문제가 있는 것으로 보인다.
- train에서 metric과 val 함수를 뜯어보기  
  
---  
- 현재 보고 있는 부분  
  -  def _train_a_batch(self, batch)   
    
---
- 문제가 있을 수 있는 부분
  - 데이터 사이즈 혹은 n_a의 사이즈 문제 -> paper와 최대한 유사한 세팅으로 확인 가능
  - metric의 잘못된 정의 -> 뜯어보는 중 Accuracy를 이용하고 있음
    - Accuracy가 파일 내에서 정의된 함수였는데, 이를 다른 걸로 바꿔야할 듯 하다. 
    - 일단 Nbsloss가 존재하니까 이걸 먼저 뜯어보자.
  - world size 설정(이 부분은 분산컴퓨팅에 대한 이해를 하고 설정을 건드려봐야할 듯 함)

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

* 아직 더 뜯어봐야하는 부분
  - NbsCls in nbsnet.py : 마지막에 레이어를 추가하는 부분. 여기서 alpha가 int가 아닐 때 시행하는 부분이 이해안됨.
  - 
