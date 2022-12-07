# How to run this code?

1. 데이터를 전처리하기 위해 `preprocess.py`를 먼저 실행시켜주세요. 

2. 학습할 모델과 그에 맞는 데이터셋 로더를 선택해주세요. 

If you want to run the MonaCoBERT, you have to use pid_loaders. For example,

```
python train.py --model_fn model.pth --model_name monacobert --dataset_name icecream_pid
```

If you want to run the MonaCoBERT_CTT, you have to use pid_diff_loaders. For example,

```
python train.py --model_fn model.pth --model_name monacobert_ctt --dataset_name icecream_pid_diff
```

# Result 
(with max_seq_len: `32`, learning_rate `0.001`, 나머지는 default setting과 동일)

pid model
| Model | monacobert | 
| ---- | ---- | 
| AUC | 0.8251 | 

pid + diff model
| Model | monacobert_ctt | bert_ctt | monabert_ctt | cobert_ctt |
| ---- | ---- | ---- | ---- | ---- | 
| AUC | **0.8280** | 0.8181 | 0.8254 | 0.8252 |