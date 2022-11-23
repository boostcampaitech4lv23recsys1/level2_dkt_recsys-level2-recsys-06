import numpy as np
import pandas as pd

def split_data(train_df, test_df, train_index, valid_index):
    # 유의해야할 점은 train_index, valid_index는 Dataframe에서 중복없는 userID를 통해 split
    # 이 함수에서 index를 나눠주는 과정까지 작성하려했으나, kfold의 경우 for문을 거쳐야하기 때문에 따로 빼둠.
    train_data = train_df[train_df['userID'].isin(train_index)]
    valid_data = train_df[train_df['userID'].isin(valid_index)]

    #validation의 마지막 문항이 아닌 데이터, 마지막 문항 데이터로 나눠 각각 처리한다.
    valid_for_train = valid_data[valid_data['userID'] == valid_data['userID'].shift(-1)]
    validset = valid_data[valid_data['userID'] != valid_data['userID'].shift(-1)]

    #test의 마지막 문항이 아닌 데이터, 마지막 문항 데이터로 나눠 각각 처리한다.
    test_for_train = test_df[test_df['userID'] == test_df['userID'].shift(-1)]
    testset = test_df[test_df['userID'] != test_df['userID'].shift(-1)]

    trainset = pd.concat([train_data, valid_for_train, test_for_train], axis = 0)

    return trainset, validset, testset