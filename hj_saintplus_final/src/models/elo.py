import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def elo(df,tag): # tag= assessmentItemID / KnowledgeTag / testId  => 셋 중 "실력"을 예측하고 싶은 녀석을 넣어준다.
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):  # theta는 대상의 Elo rating값
        '''
        is_good_answer : 맞췄으면 1, 틀렸으면 0
        beta : 
        left_asymptote : 
        theta : 유저의 점수  (유저의 경우 문제를 잘 풀수록 높음, 테스트의 경우 유저가 문제를 못 풀수록 높음)
        nb_previous_answers : 
        
        '''
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):  # beta는
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def learning_rate_theta(nb_answers):  #
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote):  # 승률
        '''
        
        '''
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def estimate_parameters(answers_df, granularity_feature_name=tag):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}  # 해당 feature의 unique값을 순회하면서 각각 beta와 nb_answer의 초기값을 만들어줌
            for granularity_feature_value in np.unique(
                answers_df[granularity_feature_name]
            )
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        print("Elo estimating start...", flush=True)
        for student_id, item_id, left_asymptote, answered_correctly in tqdm(
            zip(
                answers_df.userID.values,
                answers_df[granularity_feature_name].values,
                answers_df.left_asymptote.values, # 그래서 얘가 뭔데
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

        print(f"Elo estimation completed.")
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

    df["elo"] = prob

    return df