import random
import time
from datetime import datetime
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def get_time_lag(df):
    """
    Compute time_lag feature, same task_container_id shared same timestamp for each user
    """
    time_dict = {}
    time_lag = np.zeros(len(df), dtype=np.float32)
    for idx, row in enumerate(df[["userID", "Timestamp", "testId"]].values):
        row[1] = time.mktime(datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S').timetuple())
        if row[0] not in time_dict:
            time_lag[idx] = 0
            time_dict[row[0]] = [row[1], row[2], 0] # last_timestamp, last_task_container_id, last_lagtime
        else:
            if row[2] == time_dict[row[0]][1]:
                time_lag[idx] = time_dict[row[0]][2]
            else:
                time_lag[idx] = row[1] - time_dict[row[0]][0]
                time_dict[row[0]][0] = row[1]
                time_dict[row[0]][1] = row[2]
                time_dict[row[0]][2] = time_lag[idx]

    df["time_lag"] = time_lag/1000/60 # convert to miniute
    df["time_lag"] = df["time_lag"].clip(0, 1440) # clip to 1440 miniute which is one day => 문제푼지 하루가 지났다면 1440(60*24)로 만들어주고, 아니라면 그대로, 0보다 작다면 0으로 만들어줌
    return time_dict

def duration(df):
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['months'] = df['Timestamp'].dt.month
    df['days'] = df['Timestamp'].dt.day
    df['ts'] = df['Timestamp'].map(pd.Timestamp.timestamp)
    df['prev_ts'] = df.groupby(['userID', 'testId', 'months','days'])['ts'].shift(1)
    df["prev_ts"] = df["prev_ts"].fillna(0)
    df["elapsed"] = np.where(df["prev_ts"] == 0, 0, df["ts"] - df["prev_ts"])

    indexes = df[df['elapsed'] > 1200].index
    df.loc[indexes, 'elapsed'] = 1200
    df = df.drop(['months','days','ts','prev_ts'],axis='columns')
    return df

def make_assess_ratio(df):
    ratio_dict = defaultdict(float)
    grouped_dict = dict(df.groupby('assessmentItemID')['answerCode'].value_counts())
    assess_keys = list(set([x[0] for x in grouped_dict.keys()]))
    for key in assess_keys:
        if grouped_dict.get((key, 1)):
            right = grouped_dict[(key, 1)]
        else:
            right=0
        if grouped_dict.get((key, 0)):
            wrong = grouped_dict[(key, 0)]
        else:
            wrong = 0
        ratio = right / (right + wrong)
        ratio_dict[key] = ratio

    df['assessmentItemAverage'] = df['assessmentItemID'].map(ratio_dict)
    return df

def make_user_ratio(df):
    ratio_dict = defaultdict(float)
    grouped_dict = dict(df.groupby('userID')['answerCode'].value_counts())
    user_keys = list(set([x[0] for x in grouped_dict.keys()]))
    for key in user_keys:
        if grouped_dict.get((key, 1)):
            right = grouped_dict[(key, 1)]
        else:
            right=0
        if grouped_dict.get((key, 0)):
            wrong = grouped_dict[(key, 0)]
        else:
            wrong = 0
        ratio = right / (right + wrong)
        ratio_dict[key] = ratio

    df['UserAverage'] = df['userID'].map(ratio_dict)
    return df

def elo(df,tag): # tag= assessmentItemID / KnowledgeTag / testId
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def estimate_parameters(answers_df, granularity_feature_name=tag):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(
                answers_df[granularity_feature_name]
            )
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        print("Parameter estimation is starting...", flush=True)

        for student_id, item_id, left_asymptote, answered_correctly in tqdm(
            zip(
                answers_df.userID.values,
                answers_df[granularity_feature_name].values,
                answers_df.left_asymptote.values,
                answers_df.answerCode.values,
            ),
            total=len(answers_df),
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]
						
						# item_parameter에 문제에 대한 난이도를 수정
            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                item_parameters[item_id]["nb_answers"],
            )
						# student_parameter에 user가 느끼는 문제에 대한 난이도를 수정
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                student_parameters[student_id]["nb_answers"],
            )

            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1

        print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
        return student_parameters, item_parameters

    def gou_func(theta, beta): # elo feature을 0과 1사이의 값으로 바꿔주기 위해 사용
        return 1 / (1 + np.exp(-(theta - beta)))
    df["left_asymptote"] = 0
    print(f"Dataset of shape {df.shape}")
    print(f"Columns are {list(df.columns)}")
    student_parameters, item_parameters = estimate_parameters(df)
    prob = [
        gou_func(student_parameters[student]["theta"], item_parameters[item]["beta"])
        for student, item in zip(df['userID'].values, df[tag].values)
    ]
    df['elo'] = prob
    return df