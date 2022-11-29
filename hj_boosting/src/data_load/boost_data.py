from ..models.elo import elo

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder



def feature_engineering(df):
    # 유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    # 유저의 지금까지의 정답률 = 이 시점 직전까지 푼 문제의 개수/ 이 시점 직전까지 푼 전체 개수
    print('---------------feature engineering start---------------')
    df = df.copy()
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    
    # 시험지(문제)의 첫 세 자리 => 난이도와 상관관계를 가짐
    df['test_front'] = df['testId'].str[:3]
    
    # Timestamp 사용해서 month, day, ts를 추가
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['ts'] = df['Timestamp'].map(pd.Timestamp.timestamp)
    
    # 푸는 데 걸린 시간 추가
    df['prev_ts'] = df.groupby(['userID', 'testId', 'month','day'])['ts'].shift(1)
    df["prev_ts"] = df["prev_ts"].fillna(0)
    df["duration"] = np.where(df["prev_ts"] == 0, 0, df["ts"] - df["prev_ts"])
    duration_exceed = df[df['duration'] > 1200].index
    # 푸는 시간이 20분 넘어갔으면 20으로 고정함
    df.loc[duration_exceed, 'duration'] = 1200

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_i = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
    correct_i.columns = ["item_mean", "item_sum"]
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    # correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    # correct_k.columns = ["tag_mean", 'tag_sum']

    df = pd.merge(df, correct_i, on=['assessmentItemID'], how="left")
    df = pd.merge(df, correct_t, on=['testId'], how="left")
    # df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    df = df.drop(columns=['user_correct_answer', 'user_total_answer', 'ts', 'prev_ts'])
    
    return df




def process_boost_data(df):
    f_df = feature_engineering(df)
    
    #사용할 피쳐 선택하기
    features_and_target = ['userID', 'assessmentItemID', 'KnowledgeTag', 'user_acc', 'test_front', 'month', 'duration', 'item_mean', 'item_sum', 'test_mean', 'test_sum', 'answerCode']
    f_df = f_df[features_and_target]
    
    # 범주형 변수들 라벨 인코딩
    user_encoder = LabelEncoder()
    quiz_encoder = LabelEncoder()
    tag_encoder = LabelEncoder()
    test_front_encoder = LabelEncoder()

    user_encoder.fit(f_df['userID'])
    quiz_encoder.fit(f_df['assessmentItemID'])
    tag_encoder.fit(f_df['KnowledgeTag'])
    test_front_encoder.fit(f_df['test_front'])

    f_df.loc[:, 'userID'] = user_encoder.transform(f_df['userID'])
    f_df.loc[:, 'assessmentItemID'] = quiz_encoder.transform(f_df['assessmentItemID'])
    f_df.loc[:, 'KnowledgeTag'] = tag_encoder.transform(f_df['KnowledgeTag'])
    f_df.loc[:, 'test_front'] = test_front_encoder.transform(f_df['test_front'])

    f_df['KnowledgeTag'] = f_df['KnowledgeTag'].astype('category')
    f_df['test_front'] = f_df['test_front'].astype('category')
    f_df['month'] = f_df['month'].astype('category')

    f_df.dtypes
    
    # elo 레이팅 매기기
    for tag in ['assessmentItemID', 'test_front', 'userID']:
        f_df = elo(f_df, tag)
        f_df.rename(columns={'elo': tag+'Elo'}, inplace=True)
        f_df.drop(columns=['left_asymptote'], inplace=True)
    
    # 훈련에 사용할 train_d와 테스트에서 예측할 test_df를 분리함
    f_train_df = f_df[f_df['answerCode']!=-1]
    f_test_df = f_df[f_df['answerCode']==-1]

    features = ['userID', 'assessmentItemID', 'KnowledgeTag', 'test_front', 'month', 'duration', 'item_sum', 'test_sum', 'assessmentItemIDElo', 'test_frontElo', 'userIDElo']
  
    return f_train_df, f_test_df, features



def boost_data_load(args):
    train_df = pd.read_csv(args.DATA_PATH + 'train_data.csv', parse_dates=['Timestamp'])
    test_df = pd.read_csv(args.DATA_PATH + 'test_data.csv', parse_dates=['Timestamp'])

    df = pd.concat([train_df, test_df])
    df = df.sort_values(by=['userID','Timestamp'])
    
    f_train_df, f_test_df, features = process_boost_data(df)

    data = {
        'train_data' : f_train_df,
        'test_data' : f_test_df,
        'features' : features
    }

    return data
