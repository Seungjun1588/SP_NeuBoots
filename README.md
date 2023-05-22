# SP_NeuBoots

*일반적인 linear regression에서의 성능을 1차적으로 확인하고 있음.*
 
일단 수정한 모델에 대한 디버깅은 끝나서 결과는 나온다. 허나 metric에 대한 문제가 있는 것으로 보인다.
- train에서 metric과 val 함수를 뜯어보기  
  
  
- 문제가 있을 수 있는 부분
  - 데이터 사이즈 혹은 n_a의 사이즈 문제
  - metric의 잘못된 정의
