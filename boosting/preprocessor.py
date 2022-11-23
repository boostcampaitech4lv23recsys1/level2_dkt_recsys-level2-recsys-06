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


class Preprocessor():
    def __init__(self, args):
        self.args = args
        self.train_data_path = os.path.join(self.args.data_path, "train_data.csv")
        self.test_data_path = os.path.join(self.args.data_path, "test_data.csv")

        self.assessment_encoder = LabelEncoder()
        self.testid_encoder = LabelEncoder()
        self.knowledgetag_encoder = LabelEncoder()
        self.duration_scaler = StandardScaler()
        

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
        df['assessmentItemID'] = self.assessment_encoder.transform(df['assessmentItemID'])
        df['testId'] = self.testid_encoder.transform(df['testId'])
        df['KnowledgeTag'] = self.knowledgetag_encoder.transform(df['KnowledgeTag'])
        df['duration'] = self.duration_scaler.normalize(df['duration'])

        return df

    def preprocess_dataset(self):
        train_df, test_df= self._load_dataset()
        
        print(f"[FEATURE ENGINEERING]")
        print("starting to make duration feature")
        train_df = self.make_duration_feature(train_df)
        test_df = self.make_duration_feature(test_df)

        print("starting to make answer ratio feature")
        train_df, test_df = self.make_assess_ratio(train_df, test_df)

        using_features = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 'duration', 'ratio', 'answerCode']
        categ_features = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag']
        train_df = train_df[using_features]
        test_df = test_df[using_features]

        self.assessment_encoder.fit(train_df['assessmentItemID'])
        self.testid_encoder.fit(train_df['testId'])
        self.knowledgetag_encoder.fit(train_df['KnowledgeTag'])
        self.duration_scaler.build(train_df['duration'])

        train_df = self.encode_categ_features(train_df)
        test_df = self.encode_categ_features(test_df)


        return train_df, test_df, categ_features


