import time
import pickle
import numpy as np
import pandas as pd
from utils import get_time_lag
from utils import duration,make_assess_ratio,make_user_ratio

"""
data is from kaggler: tito's strategy
Because this is a time series competition, training and validation dataset should be split by time.
If we only use last several rows for each user as validation, we'll probably focusing too much on light user.
But timestamp feature in original data only specified elapsed time since the user's first event.
We have no idea what's actual time in real world!
tito use a strategy that it first finds maximum timestamp over all users and choose it as upper bound.
Then for each user's own maximum timestamp, Max_TimeStamp subtracts this timestamp to get a interval that
when user might start his/her first event. Finally, random select a time within this interval to get
a virtual start time and add to timestamp feature for each user.
Sort it by virtual timestamp we could then eazily split train/validation by time.
This program will produce three group files:
1. training
2. validation
3. inference
and one dictionary time_dict for inference
"""

data_type = {
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "int8"
}

def indexing(tag,df):
    tag_indexing = {v:k for k,v in enumerate(df[tag].unique())}
    df[tag] = df[tag].map(tag_indexing)
    return df

def pre_process_train(train_path,test_path, ques_path, row_start=30e6, num_rows=30e6, split_ratio=0.9, seq_len=100):
    print("Start pre-process")
    t_s = time.time()

    # Features = ["timestamp", "user_id", "content_id", "content_type_id", "task_container_id", "user_answer",
    #            "answered_correctly", "prior_question_elapsed_time", "prior_question_had_explanation", "viretual_time_stamp"]

    # timestamp => timestamp
    # user_id => userID
    # content_id => assessmentItemID
    # content_type_id => KnowledgeTag
    # task_container_id => testId
    # user_answer => X
    # user_correctly => answerCode
    # prior_questio_elapsed_time => elapsed
    # prior_question_had_explanation => X
    # viretual_time_stamp => X
    train_df = pd.read_csv(train_path)
    test_df=pd.read_csv(test_path)
    train_df = pd.concat([train_df,test_df]).reset_index(drop=True)
    train_df.index = train_df.index.astype('uint32')
    # get time_lag feature
    print("Start compute time_lag")
    time_dict = get_time_lag(train_df)
    with open("time_dict95.pkl.zip", 'wb') as pick:
        pickle.dump(time_dict, pick)
    print("Complete compute time_lag")
    print("====================")
    train_df = duration(train_df)
    total_df = make_assess_ratio(train_df)
    total_df = make_user_ratio(train_df)
    # plus 1 for cat feature which starts from 0
    train_df=indexing('assessmentItemID',train_df)
    train_df=indexing('testId',train_df)
    train_df["assessmentItemID"] += 1
    train_df["testId"] += 1
    train_df["answerCode"] += 1
    # userID	assessmentItemID	answerCode	Timestamp	KnowledgeTag	elapsed	assessmentItemAverage	answerCode_mean
    # Train_features = ['userID','assessmentItemID','testId','time_lag','Timestamp','answerCode','KnowledgeTag','elapsed',]
    Train_features = ['userID', 'assessmentItemID', 'testId', 'time_lag', 'Timestamp', 'answerCode', 'KnowledgeTag',
                      'elapsed', 'assessmentItemAverage', 'UserAverage']

    if num_rows == -1:
        num_rows = train_df.shape[0]
    train_df = train_df.iloc[int(row_start):int(row_start + num_rows)]
    print(f'{num_rows} * {split_ratio} = {num_rows * split_ratio}')
    val_df = train_df[int(num_rows * split_ratio):]
    train_df = train_df[:int(num_rows * split_ratio)]

    print("Train dataframe shape after process ({}, {})/ Val dataframe shape after process({}, {})".format(
        train_df.shape[0], train_df.shape[1], val_df.shape[0], val_df.shape[1]))
    print("====================")

    # Check data balance
    num_new_user = val_df[~val_df["userID"].isin(train_df["userID"])]["userID"].nunique()
    num_new_content = val_df[~val_df["assessmentItemID"].isin(train_df["assessmentItemID"])][
        "assessmentItemID"].nunique()
    train_content_id = train_df["assessmentItemID"].nunique()
    train_correct = train_df["answerCode"].mean() - 1
    val_correct = val_df["answerCode"].mean() - 1

    print("Number of new users {}/ Number of new contents {}".format(num_new_user, num_new_content))
    print("Number of content_id {}".format(train_content_id))
    print("train correctness {:.3f}/val correctness {:.3f}".format(train_correct, val_correct))
    print("====================")

    print("Start train and Val grouping")
    train_group = train_df[Train_features].groupby("userID").apply(lambda df: (
        df["assessmentItemID"].values,
        df["testId"].values,
        df['time_lag'].values,
        df["elapsed"].values,
        df['assessmentItemAverage'].values,
        df['UserAverage'].values,
        df["answerCode"].values
    ))
    with open("train_group95.pkl.zip", 'wb') as pick:
        pickle.dump(train_group, pick)
    del train_group, train_df

    val_group = val_df[Train_features].groupby("userID").apply(lambda df: (
        df["assessmentItemID"].values,
        df["testId"].values,
        df['time_lag'].values,
        df["elapsed"].values,
        df['assessmentItemAverage'].values,
        df['UserAverage'].values,
        df["answerCode"].values
    ))
    with open("val_group95.pkl.zip", 'wb') as pick:
        pickle.dump(val_group, pick)
    print("Complete pre-process, execution time {:.2f} s".format(time.time() - t_s))


def pre_process_validation(train_path,test_path, ques_path, row_start=30e6, num_rows=30e6, split_ratio=0.9, seq_len=100):
    print("Start pre-process")
    t_s = time.time()

    train_df = pd.read_csv(test_path)
    other_df=pd.read_csv(train_path)
    total_df = pd.concat([train_df,other_df]).reset_index(drop=True)
    train_df.index = train_df.index.astype('uint32')

    # get time_lag feature
    print("Start compute time_lag")
    time_dict = get_time_lag(train_df)
    with open("fianl_sub_time_dict.pk1.zip", 'wb') as pick:
        pickle.dump(time_dict, pick)
    print("Complete compute time_lag")
    print("====================")
    train_df=duration(train_df)
    train_df=make_assess_ratio(train_df)
    train_df=make_user_ratio(train_df)
    # plus 1 for cat feature which starts from 0
    train_df=indexing('assessmentItemID',train_df)
    train_df=indexing('testId',train_df)
    train_df["assessmentItemID"] += 1
    train_df["testId"] += 1
    train_df["answerCode"] += 1

    Train_features = ['userID', 'assessmentItemID', 'testId', 'time_lag', 'Timestamp', 'answerCode', 'KnowledgeTag',
                      'elapsed', 'assessmentItemAverage', 'UserAverage']

    if num_rows == -1:
        num_rows = train_df.shape[0]
    train_df = train_df.iloc[int(row_start):int(row_start + num_rows)]
    print(f'{num_rows} * {split_ratio} = {num_rows * split_ratio}')
    val_df = train_df[int(num_rows * split_ratio):]
    train_df = train_df[:int(num_rows * split_ratio)]

    print("Train dataframe shape after process ({}, {})/ Val dataframe shape after process({}, {})".format(
        train_df.shape[0], train_df.shape[1], val_df.shape[0], val_df.shape[1]))
    print("====================")

    # Check data balance
    num_new_user = val_df[~val_df["userID"].isin(train_df["userID"])]["userID"].nunique()
    num_new_content = val_df[~val_df["assessmentItemID"].isin(train_df["assessmentItemID"])][
        "assessmentItemID"].nunique()
    train_content_id = train_df["assessmentItemID"].nunique()
    train_correct = train_df["answerCode"].mean() - 1
    val_correct = val_df["answerCode"].mean() - 1
    print("Number of new users {}/ Number of new contents {}".format(num_new_user, num_new_content))
    print("Number of content_id {}".format(train_content_id))
    print("train correctness {:.3f}/val correctness {:.3f}".format(train_correct, val_correct))
    print("====================")

    print("Start train and Val grouping")
    train_group = train_df[Train_features].groupby("userID").apply(lambda df: (
        df["assessmentItemID"].values,
        df["testId"].values,
        df['time_lag'].values,
        df["elapsed"].values,
        df['assessmentItemAverage'].values,
        df['UserAverage'].values,
        df["answerCode"].values
    ))
    with open("fianl_sub_train_group.pkl.zip", 'wb') as pick:
        pickle.dump(train_group, pick)
    del train_group, train_df

    val_group = val_df[Train_features].groupby("userID").apply(lambda df: (
        df["assessmentItemID"].values,
        df["testId"].values,
        df['time_lag'].values,
        df["elapsed"].values,
        df['assessmentItemAverage'].values,
        df['UserAverage'].values,
        df["answerCode"].values
    ))
    with open("fianl_sub_val_group.pkl.zip", 'wb') as pick:
        pickle.dump(val_group, pick)
    print("Complete pre-process, execution time {:.2f} s".format(time.time() - t_s))


if __name__ == "__main__":
    train_path = '/opt/ml/input/data/train_data.csv'
    test_path = '/opt/ml/input/data/test_data.csv'
    ques_path = ''
    pre_process_train(train_path,test_path, ques_path, 0, -1, 0.95)
    pre_process_validation(train_path,test_path,ques_path,0,-1,0)