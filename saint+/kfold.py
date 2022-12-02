import os

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from pre_process import kfold_preprocess,pre_process_validation,feature_engineering
from train import train
from submission import kfold_main

def userid_table(train_df,feature):
    train_df = train_df.sort_values(by=["userID", "Timestamp"], axis=0)
    group = train_df[feature].groupby("userID").apply(lambda x: (x["userID"].values, 
                            x["assessmentItemID"].values, x["testId"].values,x['time_lag'].values,x['Timestamp'].values, 
                            x["answerCode"].values,x["KnowledgeTag"].values,x["elapsed"].values,x["assessmentItemAverage"].values,x["UserAverage"].values))
    return group.values

def origin_userid_table(train_df,feature):
    train_df = train_df.sort_values(by=["userID", "Timestamp"], axis=0)
    group = train_df[feature].groupby("userID").apply(lambda x: (x["userID"].values, 
                            x["assessmentItemID"].values, x["testId"].values,x['Timestamp'].values, 
                            x["answerCode"].values,x["KnowledgeTag"].values))
    return group.values

def user_indexing(train_data,train_idx,valid_idx,userby,feature):
    train_arr=[]
    valid_arr=[]
    for index in train_idx:
        temp=pd.DataFrame(userby[index],index=feature)
        temp=temp.transpose()
        train_arr.append(temp)
    for index in valid_idx:
        temp=pd.DataFrame(userby[index],index=feature)
        temp=temp.transpose()
        valid_arr.append(temp)
    train = pd.concat(train_arr,ignore_index=True,axis=0)
    valid = pd.concat(valid_arr,ignore_index=True,axis=0)
    return train,valid


submission_list = []
feature = ['userID', 'assessmentItemID', 'testId', 'time_lag', 'Timestamp', 'answerCode', 
            'KnowledgeTag', 'elapsed', 'assessmentItemAverage','UserAverage','elo'] #'UserAverage'
origin_feature=['userID', 'assessmentItemID', 'testId','Timestamp', 'answerCode','KnowledgeTag']
train_path = "/opt/ml/input/data/train_data.csv"
test_path = "/opt/ml/input/data/test_data.csv"
train_data = pd.read_csv(train_path)
test_data=pd.read_csv(test_path)
total_index=train_data['userID'].unique()
# train_data = feature_engineering(train_data)
userby = origin_userid_table(train_data,origin_feature)
pre_process_validation(test_data,train_data,'',0,-1,0)
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
for idx,(train_idx, valid_idx) in enumerate(kf.split(total_index)):
    train_df,val_df=user_indexing(train_data,train_idx,valid_idx,userby,origin_feature)
    train_df=feature_engineering(train_df)
    val_df = feature_engineering(val_df)
    kfold_preprocess(train_df,val_df,feature)
    train()
    submissions = kfold_main()
    submission_list.append(submissions)
submission_fold = pd.DataFrame()
submission_fold['id'] = np.arange(744)
submission_fold['prediction'] = np.mean(submission_list, axis=0)
submission_fold.to_csv("Saint_kfold.csv", index=False)