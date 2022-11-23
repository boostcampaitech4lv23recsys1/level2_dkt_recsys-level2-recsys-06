import pandas as pd
import numpy as np
import os

dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}
class Preprocessor():
    def __init__(self, args):
        self.args = args
        self.train_data_path = os.path.join(self.args.data_path, "train_data.csv")
        self.test_data_path = os.path.join(self.args.data_path, "test_data.csv")
        self.sub_data_path = os.path.join(self.args.data_path, "sample_submission.csv")

        
    def _load_dataset(self)
        print("starting to load data: ")
        train_df = pd.read_csv(self.train_data_path, dtype = dtype, parse_dates = ['Timestamp'], low_memory = False)\
                    .sort_values(by = ['userID', 'Timestamp']).reset_index(drop=True)

        test_df = pd.read_csv(self.test_data_path, low_memory = False)\
                    .sort_values(by = ['userID', 'Timestamp']).reset_inde(drop = True)

        sub_df = pd.read_csv(self.sub_data_path, low_memory = False)\
                    .sort_values(by = ['id']).reset_index(drop = True)

        return train_df, test_df, sub_df


    def make_duration_feature(self, df):
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
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


    def preprocess_dataset(self):
        train_df, test_df, sub_df = self._load_dataset()
        
        print(f"[FEATURE ENGINEERING]")
        print("starting to make duration feature")
        train_df = self.make_duration_feature(train_df)
        test_df = self.make_duration_feature(test_df)

        print("starting to make answer ratio feature")
        train_df, test_df = self.make_assess_ratio(train_df, test_df)
        



