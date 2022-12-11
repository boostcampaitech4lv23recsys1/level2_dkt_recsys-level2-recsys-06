import pandas as pd
import numpy as np
import os


from sklearn.preprocessing import LabelEncoder

from collections import defaultdict

DTYPE = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}


class StandardScaler:
    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def build(self, train_data):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()

    def normalize(self, df):
        return (df - self.train_mean) / self.train_std


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



class Preprocessor():
    def __init__(self, args):
        self.args = args
        self.train_data_path = os.path.join(self.args.data_path, "train_data.csv")
        self.test_data_path = os.path.join(self.args.data_path, "test_data.csv")

        self.assess_encoder = LabelEncoder()
        self.testid_encoder = LabelEncoder()
        self.knowledge_encoder = LabelEncoder()

        self.duration_scaler = StandardScaler()
        self.total_time_scaler = StandardScaler()
        self.past_correct_scaler = StandardScaler()
        self.past_content_correct_scaler = StandardScaler()
        self.mean_time_scaler = StandardScaler()
        

    def _load_dataset(self):
        print("starting to load data: ")
        train_df = pd.read_csv(self.train_data_path, dtype = DTYPE, parse_dates = ['Timestamp'], low_memory = False)\
                    .sort_values(by = ['userID', 'Timestamp']).reset_index(drop=True)

        test_df = pd.read_csv(self.test_data_path, low_memory = False)\
                    .sort_values(by = ['userID', 'Timestamp']).reset_index(drop = True)


        return train_df, test_df


    def make_duration_feature(self, df):
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['months'] = df['Timestamp'].dt.month
        df['days'] = df['Timestamp'].dt.day
        df['ts'] = df['Timestamp'].map(pd.Timestamp.timestamp)
        df['prev_ts'] = df.groupby(['userID', 'testId', 'months','days'])['ts'].shift(1)
        df["prev_ts"] = df["prev_ts"].fillna(0)
        df["duration"] = np.where(df["prev_ts"] == 0, 0, df["ts"] - df["prev_ts"])

        indexes = df[df['duration'] > 1200].index
        df.loc[indexes, 'duration'] = 1200

        return df

    def make_assess_ratio(self, train_df, test_df):
        ratio_dict = defaultdict(float)
        total_df = pd.concat([train_df, test_df], axis = 0)
        total_df = total_df[total_df['answerCode'] != -1]
        grouped_dict = dict(total_df.groupby('assessmentItemID')['answerCode'].value_counts())
        grouped_dict
        assess_keys = list(set([x[0] for x in grouped_dict.keys()]))
        for key in assess_keys:
            right = grouped_dict[(key, 1)]
            wrong = grouped_dict[(key, 0)]
            ratio = right / (right + wrong)
            ratio_dict[key] = ratio

        train_df['ratio'] = train_df['assessmentItemID'].map(ratio_dict)
        test_df['ratio'] = test_df['assessmentItemID'].map(ratio_dict)

        return train_df, test_df


    def encode_categ_features(self, df):
        df['assessmentItemID'] = self.assess_encoder.transform(df['assessmentItemID'])
        df['testId'] = self.testid_encoder.transform(df['testId'])
        df['KnowledgeTag'] = self.knowledge_encoder.transform(df['KnowledgeTag'])
        df['duration'] = self.duration_scaler.normalize(df['duration'])
        df['total_used_time'] = self.total_time_scaler.normalize(df['total_used_time'])
        df['past_correct'] = self.past_correct_scaler.normalize(df['past_correct'])
        df['past_content_correct'] = self.past_content_correct_scaler.normalize(df['past_content_correct'])
        df['mean_time'] = self.mean_time_scaler.normalize(df['mean_time'])

        return df

    def preprocess_dataset(self):
        train_df, test_df= self._load_dataset()
        total_df = pd.merge([train_df, test_df], axis =0)
        train_index = list(train_df['userID'].unique())
        test_index = list(test_df['userID'].unique())

        print(f"[FEATURE ENGINEERING]")
        print("starting to make elo ratings feature")
        for tag in ['assessmentItemID', 'test_front', 'userID']:
            total_df = elo(total_df, tag)
            total_df.rename(columns={'elo': tag+'Elo'}, inplace=True)
            total_df.drop(columns=['left_asymptote'], inplace=True)
        train_df = total_df[total_df['userID'].isin(train_index)]
        test_df = total_df[total_df['userID'].isin(test_index)]

        print("starting to make mission4 features")
        print("starting to make total used time feature")
        train_df['total_used_time'] = train_df.groupby('userID')['duration'].cumsum()
        test_df['total_used_time'] = test_df.groupby('userID')['duration'].cumsum()

        print("starting to make past correct feature")
        train_df['shift'] = train_df.groupby('userID')['answerCode'].shift().fillna(0)
        train_df['past_correct'] = train_df.groupby('userID')['shift'].cumsum()
        test_df['shift'] = test_df.groupby('userID')['answerCode'].shift().fillna(0)
        test_df['past_correct'] = test_df.groupby('userID')['shift'].cumsum()

        print("starting to make past content correct feature")
        train_df['shift'] = train_df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
        train_df['past_content_correct'] = train_df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()

        test_df['shift'] = test_df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
        test_df['past_content_correct'] = test_df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()

        print("starting to make average correct feature")
        train_df['past_count'] = train_df.groupby('userID').cumcount()
        train_df['shift'] = train_df.groupby('userID')['answerCode'].shift().fillna(0)
        train_df['past_correct'] = train_df.groupby('userID')['shift'].cumsum()
        train_df['average_correct'] = (train_df['past_correct'] / train_df['past_count']).fillna(0)
        test_df['past_count'] = test_df.groupby('userID').cumcount()
        test_df['shift'] = test_df.groupby('userID')['answerCode'].shift().fillna(0)
        test_df['past_correct'] = test_df.groupby('userID')['shift'].cumsum()
        test_df['average_correct'] = (test_df['past_correct'] / test_df['past_count']).fillna(0)

        print("starting to make average content correct feature")
        train_df['past_content_count'] = train_df.groupby(['userID', 'assessmentItemID']).cumcount()
        train_df['shift'] = train_df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
        train_df['past_content_correct'] = train_df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()
        train_df['average_content_correct'] = (train_df['past_content_correct'] / train_df['past_content_count']).fillna(0)
        test_df['past_content_count'] = test_df.groupby(['userID', 'assessmentItemID']).cumcount()
        test_df['shift'] = test_df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift().fillna(0)
        test_df['past_content_correct'] = test_df.groupby(['userID', 'assessmentItemID'])['shift'].cumsum()
        test_df['average_content_correct'] = (test_df['past_content_correct'] / test_df['past_content_count']).fillna(0)

        print("starting to make mean time feature")
        train_df['mean_time'] = train_df.groupby(['userID'])['duration'].rolling(3).mean().values
        train_df['mean_time'] = train_df['mean_time'].fillna(0)
        test_df['mean_time'] = test_df.groupby(['userID'])['duration'].rolling(3).mean().values
        test_df['mean_time'] = test_df['mean_time'].fillna(0)

        print("starting to make relative time median feature")
        total_df = pd.concat([train_df, test_df], axis = 0)
        agg_df = total_df.groupby('assessmentItemID')['duration'].agg(['median'])
        agg_dict = agg_df.to_dict()

        train_df['time_median'] = train_df['assessmentItemID'].map(agg_dict['median'])
        train_df['relative_time_median'] = np.where(train_df['duration'] < train_df['time_median'], 0, 1)
        test_df['time_median'] = test_df['assessmentItemID'].map(agg_dict['median'])
        test_df['relative_time_median'] = np.where(test_df['duration'] < test_df['time_median'], 0, 1)

        print("starting to make hour feature")
        train_df['hour'] = train_df['Timestamp'].dt.hour
        test_df['hour'] = test_df['Timestamp'].dt.hour

        print("starting to make day of week feature")
        train_df['dayofweek'] = train_df['Timestamp'].dt.dayofweek
        test_df['dayofweek'] = test_df['Timestamp'].dt.dayofweek

        print("starting to make duration feature")
        train_df = self.make_duration_feature(train_df)
        test_df = self.make_duration_feature(test_df)

        print("starting to make answer ratio feature")
        train_df, test_df = self.make_assess_ratio(train_df, test_df)

        self.assess_encoder.fit(train_df['assessmentItemID'])
        self.testid_encoder.fit(train_df['testId'])
        self.knowledge_encoder.fit(train_df['KnowledgeTag'])

        self.duration_scaler.build(train_df['duration'])
        self.total_time_scaler.build(train_df['total_used_time'])
        self.past_correct_scaler.build(train_df['past_correct'])
        self.past_content_correct_scaler.build(train_df['past_content_correct'])
        self.mean_time_scaler.build(train_df['mean_time'])

        train_df = self.encode_categ_features(train_df)
        test_df = self.encode_categ_features(test_df)

        using_features = ['userID', 'assessmentItemID', 'KnowledgeTag', 'relative_time_median', 'duration', 'userIDElo', 'assessmentItemIDElo', 'past_correct', 'average_correct',  'mean_time', 'answerCode']
        categ_features = ['userID', 'assessmentItemID', 'KnowledgeTag', 'relative_time_median']
        train_df = train_df[using_features]
        test_df = test_df[using_features]

        return train_df, test_df, categ_features


