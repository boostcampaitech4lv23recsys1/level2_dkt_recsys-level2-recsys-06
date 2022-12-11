# 라이브러리 설치

```

pip install -r requirements.txt
pip install wandb  #wandb 사용시


```


# 파일
- src
  - criterion.py : loss 계산 모듈 정의
  - dataloder.py : 데이터 로드 및 전처리 함수 정의
  - layer.py : layer 정의
  - metric.py : 평가지표 정의
  - model.py : 시퀀스 모델 정의 
  - optimizer.py : optimizer 설정 파일
  - scheduler.py : scheduler 설정 파일
  - trainer.py : 모델 build, train, inference 관련 코어 로직 정의 
  - utils.py : seed 설정
- args.py : 설정 파일
- inference.py : 시나리오에 따라 학습된 모델을 불러 테스트 데이터의 추론값을 계산하는 스크립트
- train.py : 시나리오에 따라 데이터를 불러 모델을 학습하는 스크립트
- sweep.yaml : wandb sweep으로 하이퍼파라미터 튜닝시 사용







# 사용 시나리오

- 상위 폴더에 존재하는 preprocess/assess_ratio.py 파일을 실행하여 pickle 파일 생성
- args.py 수정 : 데이터 파일/출력 파일 경로 설정 등
  - model arguments : **lstm, lstmattn, bert, lastqt, gru_lastquery**
- train.py 실행 : 데이터 학습 수행 및 모델 저장
- inference.py 실행 : 저장된 모델 로드 및 테스트 데이터 추론 수행
