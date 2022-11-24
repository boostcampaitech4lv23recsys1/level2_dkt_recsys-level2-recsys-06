import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

# def assessment_category(df):
#     assess_category = []
#     assess_number = []
#     for work in df['assessmentItemID']:
#         assess_category.append(work[3])
#         assess_number.append(work[-3:])
#     df['firstId']=assess_category
#     df['lastId']=assess_number
#     return df

def duration(test_df):
    test_df = test_df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    test_df['months'] = test_df['Timestamp'].dt.month
    test_df['days'] = test_df['Timestamp'].dt.day
    test_df['ts'] = test_df['Timestamp'].map(pd.Timestamp.timestamp)
    test_df['prev_ts'] = test_df.groupby(['userID', 'testId', 'months','days'])['ts'].shift(1)
    test_df["prev_ts"] = test_df["prev_ts"].fillna(0)
    test_df["duration"] = np.where(test_df["prev_ts"] == 0, 0, test_df["ts"] - test_df["prev_ts"])

    indexes = test_df[test_df['duration'] > 1200].index
    test_df.loc[indexes, 'duration'] = 1200
    test_df = test_df.drop(['months','days','ts','prev_ts'],axis='columns')
    return test_df

# 시드값 고정
seed = 9
random.seed(seed)
np.random.seed(seed)

def data_processing(train,fm_test,test,sub):
    ids = pd.concat([train['userID'], fm_test['userID']]).unique()
    train = duration(train)
    fm_test = duration(fm_test)
    test = duration(test)
    # train = assessment_category(train)
    # fm_test = assessment_category(fm_test)
    # test = assessment_category(test)

    idx2user = {idx:id for idx, id in enumerate(ids)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    train['userID'] = train['userID'].map(user2idx)
    test['userID'] = test['userID'].map(user2idx)
    sub['userID'] = sub['userID'].map(user2idx)
    context_df = pd.concat([train,fm_test]).reset_index(drop=True)
    train_df = train.drop(labels='Timestamp',axis=1)
    test_df = test.drop(labels='Timestamp',axis=1)

    #assessmentID indexing
    assess_indexing = {v:k for k,v in enumerate(context_df['assessmentItemID'].unique())}
    train_df['assessmentItemID'] = train_df['assessmentItemID'].map(assess_indexing)
    test_df['assessmentItemID'] = test_df['assessmentItemID'].map(assess_indexing)


    # testId indexing
    testid_indexing = {v:k for k,v in enumerate(context_df['testId'].unique())}
    train_df['testId'] = train_df['testId'].map(testid_indexing)
    test_df['testId'] = test_df['testId'].map(testid_indexing)

    # knowledgeTage indexing
    know_indexing = {v:k for k,v in enumerate(context_df['KnowledgeTag'].unique())}
    train_df['KnowledgeTag'] = train_df['KnowledgeTag'].map(know_indexing)
    test_df['KnowledgeTag'] = test_df['KnowledgeTag'].map(know_indexing)

    # #firstId indexing
    # firstid_indexing = {v:k for k,v in enumerate(context_df['firstId'].unique())}
    # train_df['firstId'] = train_df['firstId'].map(firstid_indexing)
    # test_df['firstId'] = test_df['firstId'].map(firstid_indexing)

    # #lastId indexing
    # lastid_indexing = {v:k for k,v in enumerate(context_df['lastId'].unique())}
    # train_df['lastId'] = train_df['lastId'].map(lastid_indexing)
    # test_df['lastId'] = test_df['lastId'].map(lastid_indexing)

    # duration indexing
    duration_indexing = {v:k for k,v in enumerate(context_df['duration'].unique())}
    train_df['duration'] = train_df['duration'].map(duration_indexing)
    test_df['duration'] = test_df['duration'].map(duration_indexing)

    # 필드 차원 수 정해주기
    field_dim = np.array([len(user2idx),len(assess_indexing),len(testid_indexing),len(know_indexing),len(duration_indexing)], dtype=np.uint32) # len(firstid_indexing),len(lastid_indexing),

    # 나중에 인덱싱한거 다시 되돌리기 용 및 기타 데이터 다 저장해서 넘기기 ~ data['train'] 이런식으로 조회 및 타 데이터 추가 가능하게
    data = {
            'train' : train_df,
            'test' : test_df.drop(['answerCode'], axis=1),
            'sub':sub,
            'idx2user':idx2user,
            'user2idx':user2idx,
            'field_dim' : field_dim   
            }

    return data 

def context_data_split(data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['answerCode'], axis=1),
                                                        data['train']['answerCode'],
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def stratified_kfold(data,n):
    skf = StratifiedKFold(n_splits= 5, shuffle=True, random_state=seed)
    counts = 0
    for train_index, valid_index in skf.split(data['train'].drop(['answerCode'], axis=1),data['train']['answerCode']):
        if counts == n:
            data['X_train'], data['y_train'] = data['train'].drop(['answerCode'], axis=1).loc[train_index], data['train']['answerCode'].loc[train_index]
            data['X_valid'], data['y_valid'] = data['train'].drop(['answerCode'], axis=1).loc[valid_index], data['train']['answerCode'].loc[valid_index]
            break
        else:
            counts += 1
    return data

def context_data_loader(data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1024, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    return data