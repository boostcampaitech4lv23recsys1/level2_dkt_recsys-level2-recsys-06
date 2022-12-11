import os
import random
import time
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle

import torch
import tqdm
from sklearn.preprocessing import LabelEncoder
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
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "lastid"]
        self.args.USE_COLUMN = cate_cols
        

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

        # def convert_time(s):
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

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
        df['lastid'] = df['assessmentItemID'].apply(lambda x:x[-3:])
        return df

    def load_data_from_file(self, file_name, is_train=True):
        stime = time.time()
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df, is_train)
        print(f"[FEATURE ENGINEERIN RESULT]\n {df.sample(3)} \n\n")
        df = self.__preprocessing(df, is_train)


        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy")))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, "testId_classes.npy")))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy")))
        self.args.n_lastid = len(np.load(os.path.join(self.args.asset_dir, "lastid_classes.npy")))


        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        df['check_userid'] = df['userID'].shift(-1)
        df['check'] = np.where(df['userID'] != df['check_userid'], 1, 0) #userid & testid 같이 합칠 때 마지막껏만 뺄 수 있게
        columns = ["userID", "assessmentItemID", "testId", "KnowledgeTag", 'duration', 'assess_ratio', "answerCode", 'check','lastid']

        self.args.n_embedding_layers = []       # 나중에 사용할 떄 embedding key들을 저장

        for val in self.args.USE_COLUMN:
            self.args.n_embedding_layers.append(len(np.load(os.path.join(self.args.asset_dir, val+'_classes.npy'))))
        print(f"[DATAFRAME RESULT]\n {df.sample(3)} \n\n")


        # TODO column을 변경할거면 여기서 변경할 수 있다.
        if self.args.group_mode == 'userid':
            group = (df[columns].groupby("userID").apply(lambda x: (x["userID"].values, x["testId"].values, x["assessmentItemID"].values, x["KnowledgeTag"].values, x['duration'].values, x['assess_ratio'].values, x["answerCode"].values, x['check'].values, x["lastid"].values)))
        elif self.args.group_mode == 'userid_with_testid':
            group = (df[columns].groupby(["userID", "testId"]).apply(lambda x: (x["userID"].values, x["testId"].values, x["assessmentItemID"].values, x["KnowledgeTag"].values, x['duration'].values, x['assess_ratio'].values, x["answerCode"].values, x['check'].values)))
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

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        #TODO user id를 추가해야하나?
        userid, test, question, tag, duration, assess_ratio, correct, lastid = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[8]
        # print(f"[CORRECT IN DATA LOADER ] \n {correct}")
        
        columns = [test, question, tag, duration, assess_ratio, correct, lastid]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(columns):
                columns[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        columns.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(columns):
            columns[i] = torch.tensor(col)

        return columns

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


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

def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data