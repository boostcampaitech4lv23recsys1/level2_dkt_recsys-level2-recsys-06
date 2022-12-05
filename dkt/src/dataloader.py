import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pickle

import torch
import tqdm
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict


class StandardScaler:
    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def build(self, train_data):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()

    def normalize(self, df):
        return (df - self.train_mean) / self.train_std


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.duration_normalizer = StandardScaler()

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def concat_for_train(self, data, is_valid):
        data_for_train_list = list()
        dataset_list = list()
        if is_valid:
            for idx in range(len(data)):
                if data[idx][-1][-1] == 1:
                    dataset_list.append(idx)
                else:
                    data_for_train_list.append(idx)
        else:
            for idx in range(len(data)):
                if data[idx][-2][-1] == -1:
                    dataset_list.append(idx)
                else:
                    data_for_train_list.append(idx)
        data_for_train = data[data_for_train_list]
        dataset = data[dataset_list]

        return data_for_train, dataset


    def split_data(self, train_data, test_data, train_idx, valid_idx):
        """
        user별 random split
        """
        trainset = train_data[train_idx]
        valid_data = train_data[valid_idx]
        if self.args.group_mode == 'userid_with_testid':
            valid_for_train, validset = self.concat_for_train(valid_data, True)
            test_for_train, testset = self.concat_for_train(test_data, False)
            trainset = np.append(trainset, valid_for_train)
            trainset = np.append(trainset, test_for_train)
            # print(f"[DATA FOR TRAINSET LENGTH] valid_for_train shape: {valid_for_train.shape}, test_for_train shape: {test_for_train.shape}")
        else:
            validset = valid_data                

        return trainset, validset

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        return df

    def __feature_engineering(self, df, is_train):
        """
        Make duration feature
        """
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

        if is_train:
            self.duration_normalizer.build(df['duration'])
            df['duration'] = self.duration_normalizer.normalize(df['duration'])
        else:
            df['duration'] = self.duration_normalizer.normalize(df['duration'])

        """
        Make assess_ratio feature
        """
        with open("/opt/ml/output/asset/grouped_dict.pkl", "rb") as f:
            grouped_dict = pickle.load(f)
        
        ratio_dict = defaultdict(int)
        assess_keys = list(set([x[0] for x in grouped_dict.keys()]))
        for key in assess_keys:
            right = grouped_dict[(key, 1)]
            wrong = grouped_dict[(key, 0)]
            ratio = right / (right + wrong)
            ratio_dict[key] = ratio

        df['assess_ratio'] = df['assessmentItemID'].map(ratio_dict)
        return df

    def load_data_from_file(self, file_name, is_train=True):
        stime = time.time()
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, low_memory = False)  # , nrows=100000)
    
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        
        with open(os.path.join(self.args.asset_dir, "encoder.pkl"), 'rb') as f:
            encoder_dict = pickle.load(f)
        
        self.args.n_questions = len(encoder_dict['assess_encoder'].classes_)
        self.args.n_test = len(encoder_dict['testid_encoder'].classes_)
        self.args.n_tag = len(encoder_dict['knowledge_encoder'].classes_)

        df['check_userid'] = df['userID'].shift(-1)
        df['check'] = np.where(df['userID'] != df['check_userid'], 1, 0)

        columns = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 'relative_time_median', 'hour', 'dayofweek', \
                    'duration', 'userIDElo', 'assessmentItemIDElo', 'testIdElo', 'KnowledgeTagElo', \
                    'past_correct', 'average_correct',  'mean_time', 'answerCode']
        # 13개의 feature
        print(f"[DATAFRAME RESULT]\n {df.sample(3)} \n\n")


        # TODO column을 변경할거면 여기서 변경할 수 있다.
        if self.args.group_mode == 'userid':
            # group = (df[columns].groupby("userID").apply(lambda x: (x["userID"].values, x["testId"].values, x["assessmentItemID"].values, x["KnowledgeTag"].values, x['duration'].values, x['assess_ratio'].values, x["answerCode"].values, x['check'].values)))
            group = (df[columns].groupby('userID').apply(lambda x: (x['userID'].values, x['assessmentItemID'].values, x['testId'].values, x['KnowledgeTag'].values, x['relative_time_median'].values, x['hour'].values, x['dayofweek'].values\
                                                                    , x['duration'].values, x['userIDElo'].values, x['assessmentItemIDElo'].values, x['testIdElo'].values, x['KnowledgeTagElo'].values\
                                                                    , x['past_correct'].values, x['average_correct'].values, x['mean_time'].values, x['answerCode'].values)))
        elif self.args.group_mode == 'userid_with_testid':
            # group = (df[columns].groupby(["userID", "testId"]).apply(lambda x: (x["userID"].values, x["testId"].values, x["assessmentItemID"].values, x["KnowledgeTag"].values, x['duration'].values, x['assess_ratio'].values, x["answerCode"].values, x['check'].values)))
            group = (df[columns].groupby(['userID', 'testId']).apply(lambda x: (x['userID'].values, x['assessmentItemID'].values, x['testId'].values, x['KnowledgeTag'].values, x['relative_time_median'].values, x['hour'].values, x['dayofweek'].values\
                                                                    , x['duration'].values, x['userIDElo'].values, x['assessmentItemIDElo'].values, x['testIdElo'].values, x['KnowledgeTagElo'].values\
                                                                    , x['past_correct'].values, x['average_correct'].values, x['mean_time'].values, x['answerCode'].values)))
        print(f"[PROCESS TIME]: {time.time() - stime}sec \n\n\n")
        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.using_features = ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 'relative_time_median', 'hour', 'dayofweek', \
                    'duration', 'userIDElo', 'assessmentItemIDElo', 'testIdElo', 'KnowledgeTagElo', \
                    'past_correct', 'average_correct',  'mean_time', 'answerCode']

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        case = {}
        for idx, col in enumerate(self.using_features):
            case[col] = row[idx]

        if seq_len > self.args.max_seq_len:
            for idx, col in enumerate(self.using_features):
                case[col] = case[col][-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 case 목록에 포함시킴
        case['mask'] = torch.tensor(mask)

        # np.array -> torch.tensor 형변환
        for idx, col in enumerate(self.using_features):
            case[col] = torch.tensor(case[col])
        case['using_features'] = self.using_features

        return case

    def __len__(self):
        return len(self.data)


def collate(batch):
    #batch에는 list로 전달된다. 리스트의 element로는 dictionary가 존재
    using_features = batch[0]['using_features'] + ['mask']

    # col_n = len(batch[0])
    # col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0]['mask'])
    data = defaultdict(list)
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for idx, col in enumerate(using_features):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(row[col]) :] = row[col]
            data[col].append(pre_padded)

    for i, col in enumerate(using_features):
        data[col] = torch.stack(data[col])

    data['using_features'] = using_features
    return data


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
