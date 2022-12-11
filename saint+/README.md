## Saint+
![image](https://user-images.githubusercontent.com/46878756/206893969-a863c725-3b77-4f35-91bc-caa666c39483.png)

학생들의 과거이력 칼럼이 디코더 입력으로 사용되고, 디코더의 첫 번째 layer는 응답 간의 관계, 질문 작업에 소요된 시간 및 사용자가 응답한 다른 문제간의 시간 간격을 학습한다. 첫 번째 디코더의 output sequence는 query로 다음 layer에 전달한다.

직관적인 설명은 지식 경험(query)에 대한 과거 경험을 가지고 있으며, 문제 풀이(key, value)에 대해 학생들이 어떻게 수행할 것인지(weighted value vector)로 표현할 수 있다
### Get started
#### 1. original
```
cd saint+
python pre_precess.py
python train.py --parameters
python submission.py
```
#### 2. kfold
```
python kfold.py --parameters
```
### parameters
```
--saved_dir : model dir
--file_name : 'model name
--num_layers : number of multihead attention layer(default: 2)
--num_heads : number of head in one multihead attention layer(default: 3)
--d_model : dimension of embedding size(default: 64)
--max_len : maximum index for position encoding(default: 1000)
--n_questions : number of different question(default: 13523)
--n_ans : number of choice of answer(default: 2)
--seq_len : sequence length(default: 100)
--warmup_steps : warmup_steps for learning rate(default: 4000)
--dropout : dropout ratio(default: 0.1)
--epochs : number of epochs(default: 30)
--patience : patience to wait before early stopping(default: 5)
--batch_size : batch size(default: 512)
--optimizer' : optimizer(default:'adam')
--lr : learning rate(default:0.001)
```

### Performance
| type | public | private |
| :--------------------------------------------------------------------------------------:| :--------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------:
|                   original                                |                        0.8058        |                   0.8249                                |   
|                   kfold                              |                        0.8049        |                   0.8402                                |   


### Reference
https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch
