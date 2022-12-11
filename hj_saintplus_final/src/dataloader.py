import os
import random
import time
import pickle

from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder

from src.models.elo import elo

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def count_classes(self):
        self.args.n_questions = len(
        np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        self.args.n_test_front = len(
            np.load(os.path.join(self.args.asset_dir, "test_front_classes.npy"))
        )
        self.args.n_userid = len(
            np.load(os.path.join(self.args.asset_dir, "userID_classes.npy"))
        )
        self.args.n_month = len(
            np.load(os.path.join(self.args.asset_dir, "month_classes.npy"))
        )
        self.args.n_day = len(
            np.load(os.path.join(self.args.asset_dir, "day_classes.npy"))
        )
        self.args.n_hour = len(
            np.load(os.path.join(self.args.asset_dir, "hour_classes.npy"))
        )

    # user별로 이미 묶어놨기 때문에 user별로 셔플이 됨
    def split_data(self, data, ratio=0.8, shuffle=True, seed=42):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        # 범주형 피쳐 지정해주기
        cate_cols = ["assessmentItemID", "KnowledgeTag", "testId", "test_front", "userID", "month", "day", "hour"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
        # 범주형 변수에 라벨 인코더를 더해주는 과정
        for col in cate_cols:
            le = LabelEncoder()
            if is_train: # 만약 훈련 중이면
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                # 라벨들의 도메인값을 .npy 파일로 디스크에 저장
                self.__save_labels(le, col)
            else:  # 훈련이 아니면 이미 저장되어있는 걸 불러오기
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            # string으로 바꿔준 다음 인코더로 바꿔준다
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
    
            df[col] = test
    
        return df

    def __feature_engineering(self, df, is_train=True):
        df = df.copy()
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

        # 시험지(문제)의 첫 세 자리 => 난이도와 상관관계를 가짐
        df['test_front'] = df['testId'].str[:3]

        # Timestamp 사용해서 month, day, ts를 추가
        df['month'] = df['Timestamp'].dt.month
        df['day'] = df['Timestamp'].dt.day
        df['hour'] = df['Timestamp'].dt.hour
        df['ts'] = df['Timestamp'].map(pd.Timestamp.timestamp)

        # 푸는 데 걸린 시간 추가
        df['prev_ts'] = df.groupby(['userID', 'testId', 'month','day'])['ts'].shift(1)
        df["prev_ts"] = df["prev_ts"].fillna(0)
        df["duration"] = np.where(df["prev_ts"] == 0, 10, df["ts"] - df["prev_ts"])
        duration_exceed = df[df['duration'] > 1200].index
        # 푸는 시간이 20분 넘어갔으면 20으로 고정함
        df.loc[duration_exceed, 'duration'] = 1200

        # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
        # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
        correct_i = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
        correct_i.columns = ["item_mean", "item_sum"]
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']

        df = pd.merge(df, correct_i, on=['assessmentItemID'], how="left")
        df = pd.merge(df, correct_t, on=['testId'], how="left")

        for tag in ['assessmentItemID', 'test_front', 'userID', 'KnowledgeTag']:
            df = elo(df, tag)
            df.rename(columns={'elo': tag+'Elo'}, inplace=True)
            df.drop(columns=['left_asymptote'], inplace=True)
        
        df = df.drop(columns=['user_correct_answer', 'ts', 'prev_ts'])
        df[['user_acc']] = df[['user_acc']].fillna(1)
        
        if is_train:
            with open("./featured_df", "wb") as file:
                pickle.dump(df, file)

        return df


    def load_data_from_file(self, file_name, is_train=True):
        if os.path.exists('./featured_df'):
            with open('./featured_df', "rb") as file:
                df = pickle.load(file)

        else:
            train_path = os.path.join(self.args.data_dir, 'train_data.csv')
            test_path = os.path.join(self.args.data_dir, 'test_data.csv')
            tr_df = pd.read_csv(train_path, parse_dates=['Timestamp'])
            te_df = pd.read_csv(test_path, parse_dates=['Timestamp'])
            tr_df['is_tr'] = 1
            te_df['is_tr'] = 0
            df = pd.concat([tr_df, te_df])

            df = self.__feature_engineering(df)


        train_df = df[df['answerCode']!=-1]
        test_df = df[df['is_tr']==0]
    
        breakpoint()

        features_and_target = [
            'assessmentItemID', 'KnowledgeTag', 'testId', 'test_front', 'item_mean', 'item_sum', 
            'test_mean', 'test_sum', 'assessmentItemIDElo',
            'userIDElo', 'userID', 'user_acc', 'user_total_answer', 'test_frontElo',
             'month', 'day', 'hour', 'duration', 'answerCode'
             ] 
        
        train_df = self.__preprocessing(train_df[features_and_target], is_train=True)
        test_df = self.__preprocessing(test_df[features_and_target], is_train=False)
        
        columns = [
            'assessmentItemID', 'KnowledgeTag', 'testId', 'test_front', 'item_mean', 'item_sum',  
            'test_mean', 'test_sum', 'assessmentItemIDElo',
            'userIDElo', 'userID', 'user_acc', 'user_total_answer', 'test_frontElo',
             'month', 'day', 'hour', 'duration', 'answerCode'
             ]
             

        # userID별로 그룹지어서 피쳐를 순회하면서 각각의 피쳐값을 넣어준 시리즈를 반환함
        group = (
            train_df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    *(r[column].values for column in columns),
                )
            )
        )

        with open("./train_data", "wb") as file:
            pickle.dump(group.values, file)

        te_group = (
            test_df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    *(r[column].values for column in columns),
                )
            )
        ) 
              
        with open("./test_data", "wb") as file:
            pickle.dump(te_group.values, file)

        return group.values, te_group.values

    def load_data(self, file_name):
        is_ready = 0
        if os.path.exists('./train_data'):
            with open('./train_data', "rb") as trfile:
                self.train_data = pickle.load(trfile)
            is_ready +=1
        
        if os.path.exists('./test_data'):
            with open('./test_data', "rb") as tefile:
                self.test_data = pickle.load(tefile)
            is_ready +=1

        if is_ready !=2:
            self.train_data, self.test_data = self.load_data_from_file(file_name, is_train=False)
        
        self.count_classes()
        
        

class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    # __getitem__ 메소드는 데이터에서 인덱스로 레코드를 불러올 때 사용된다. 여기서 index는 유저의 id.
    def __getitem__(self, index):
         # 레코드 하나. row 하나에는 총 18개 피쳐 + answerCode가 각 유저의 문제 풀이 개수만큼 들어있다. 
         # 즉, row의 shape은 (19, seq_len)
    
        row = self.data[index]
        
        # 각 data의 sequence length
        seq_len = len(row[0])  # 현재 유저가 푼 문제의 숫자

        # 앞서 만든 데이터 셋에서 각각 해당 유저가 푼 문제 수를 길이로 가지는 테스트번호/문제번호/태그번호/정답여부 배열
        cate_cols = [*row]

        # cate_cols = [test, question, tag, correct]

        # max seq len보다 길면 자르는 단계
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]  # 유저별로 뒤에서 max_seq_len번째 데이터부터만 남겨둔다.
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)  # 이건 뭘까? 데이터가 있는 곳에는 1을 채워둔다?
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols  # 마스크해서 보내줌

    def __len__(self):
        return len(self.data)


def collate(batch):
    col_n = len(batch[0])  # 피쳐 + 타겟 + 마스크  의 개수
    col_list = [[] for _ in range(col_n)]  # 마스킹 결과를 저장할 리스트
    max_seq_len = len(batch[0][-1])  # 마스크의 길이가 max_seq_len이므로

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

