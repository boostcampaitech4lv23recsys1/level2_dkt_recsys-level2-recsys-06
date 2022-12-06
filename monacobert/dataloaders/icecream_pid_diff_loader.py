import os
import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

DATASET_DIR = "../../data/preprocessed_df.csv"
PICKLE_DIR = "pickle/"

class ICECREAM_PID_DIFF(Dataset):
    def __init__(self, max_seq_len, config=None, dataset_dir=DATASET_DIR, pickle_dir=PICKLE_DIR) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir

        # 추가
        self.config = config
        self.pickle_dir = pickle_dir
        
        self.train_usernum = 6698   # Hard Coding
        
        if os.path.exists(os.path.join(self.pickle_dir, "diff_seqs.pkl")):
            with open(os.path.join(self.pickle_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "r_list.pkl"), "rb") as f:
                self.r_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "pid_seqs.pkl"), "rb") as f:
                self.pid_seqs = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "diff_seqs.pkl"), "rb") as f:
                self.diff_seqs = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "pid_list.pkl"), "rb") as f:
                self.pid_list = pickle.load(f)
            with open(os.path.join(self.pickle_dir, "diff_list.pkl"), "rb") as f:
                self.diff_list = pickle.load(f)

        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, \
                self.r_list, self.q2idx, self.u2idx, self.pid_seqs, \
                    self.diff_seqs, self.pid_list, self.diff_list = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        self.num_diff = self.diff_list.shape[0]

        self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs, train_add, test_add = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.diff_seqs, max_seq_len)

        self.len = len(self.q_seqs)
        self.train_len = self.train_usernum + train_add 

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.pid_seqs[index], self.diff_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, sep=',')
        # temporary change -1 to 0
        df['correct'].replace(-1,0,inplace=True)
        # df = df[(df["correct"] == 0) | (df["correct"] == 1)]

        # u_list = np.unique(df["user_id"].values)
        u_list_index = np.unique(df["user_id"].values, return_index=True)[1]
        u_list = np.array([df["user_id"].values[index] for index in sorted(u_list_index)])
        # q_list = np.unique(df["skill_id"].values)
        q_list_index = np.unique(df["skill_id"].values, return_index=True)[1]
        q_list = np.array([df["skill_id"].values[index] for index in sorted(q_list_index)])
        # r_list = np.unique(df["correct"].values)
        r_list_index = np.unique(df["correct"].values, return_index=True)[1]
        r_list = np.array([df["correct"].values[index] for index in sorted(r_list_index)])
        # pid_list = np.unique(df["item_id"].values)
        pid_list_index = np.unique(df["item_id"].values, return_index=True)[1]
        pid_list = np.array([df["item_id"].values[index] for index in sorted(pid_list_index)])

        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        pid2idx = {pid: idx for idx, pid in enumerate(pid_list)} 

        # difficult
        # diff = np.round(df.groupby('item_id')['correct'].mean() * 100)
        # diff_list = np.unique(df.groupby('item_id')['correct'].mean())

        u_idx = np.arange(int(len(u_list)))
        u_train_idx = u_idx[ :self.train_usernum]

        train_u_idx = u_train_idx[ : int(len(u_list) * self.config.train_ratio) ]
        valid_u_idx = u_train_idx[ int(len(u_list) * self.config.train_ratio) : ]
        test_idx = u_idx[self.train_usernum: ]

        q_seqs = []
        r_seqs = []
        pid_seqs = []
        #diff_seqs = []

        # for diff
        train_pid_seqs = []
        train_r_seqs = []

        for idx, u in enumerate(u_list):
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correct"].values
            pid_seq = np.array([pid2idx[pid] for pid in df_u["item_id"].values])
            #diff_seq = np.array([diff[item] for item in df_u["item_id"].values])

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            pid_seqs.append(pid_seq)
            #diff_seqs.append(diff_seq)

            if idx in train_u_idx:
                train_pid_seqs.extend(pid_seq)
                train_r_seqs.extend(r_seq)

        # train_df
        train_df = pd.DataFrame(
            zip(train_pid_seqs, train_r_seqs), 
            columns = ["pid", "r"]
            )
        # pid_diff
        train_pid_diff = np.round(train_df.groupby('pid')['r'].mean() * 100)
        # diff_list = np.unique(train_df.groupby('pid')['r'].mean()) 
        diff_list_index = np.unique(train_df.groupby('pid')['r'].mean(), return_index=True)[1]
        diff_list = np.array([train_df.groupby('pid')['r'].mean()[index] for index in sorted(diff_list_index)])
        
        # <class 'pandas.core.series.Series'>

        diff_seqs = []

        # train_pid_list = np.unique(train_pid_seqs)
        train_pid_list_index = np.unique(train_pid_seqs, return_index=True)[1]
        train_pid_list = np.array([train_pid_seqs[index] for index in sorted(train_pid_list_index)])

        for pid_seq in pid_seqs:

            pid_diff_seq = []

            for pid in pid_seq:
                if pid not in train_pid_list:
                    pid_diff_seq.append(float(75)) # <PAD>
                else:
                    pid_diff_seq.append(train_pid_diff[pid])

            diff_seqs.append(pid_diff_seq)
        
        with open(os.path.join(self.pickle_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.pickle_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.pickle_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.pickle_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.pickle_dir, "r_list.pkl"), "wb") as f:
            pickle.dump(r_list, f)
        with open(os.path.join(self.pickle_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.pickle_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)
        with open(os.path.join(self.pickle_dir, "pid_seqs.pkl"), "wb") as f:
            pickle.dump(pid_seqs, f)
        with open(os.path.join(self.pickle_dir, "diff_seqs.pkl"), "wb") as f:
            pickle.dump(diff_seqs, f)
        with open(os.path.join(self.pickle_dir, "pid_list.pkl"), "wb") as f:
            pickle.dump(pid_seqs, f)
        with open(os.path.join(self.pickle_dir, "diff_list.pkl"), "wb") as f:
            pickle.dump(diff_list, f)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx, pid_seqs, diff_seqs, pid_list, diff_list #끝에 두개 추가

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, diff_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []
        proc_diff_seqs = []

        train_add = 0
        test_add = 0
        
        for idx, (q_seq, r_seq, pid_seq, diff_seq) in enumerate(zip(q_seqs, r_seqs, pid_seqs, diff_seqs)):

            i = 0
            # while i + max_seq_len < len(q_seq): 
            #     if idx < self.train_usernum : train_add += 1
            #     else : test_add += 1
            #     proc_q_seqs.append(q_seq[i:i + max_seq_len])
            #     proc_r_seqs.append(r_seq[i:i + max_seq_len])
            #     proc_pid_seqs.append(pid_seq[i:i + max_seq_len])
            #     proc_diff_seqs.append(diff_seq[i:i + max_seq_len])

            #     i += max_seq_len

            # train set 의 경우 max_length를 기준으로 시퀀스를 잘라서 모두 넣어줌
            if idx < self.train_usernum : 
                while i + max_seq_len < len(q_seq): 
                    train_add += 1
                    proc_q_seqs.append(q_seq[i:i + max_seq_len])
                    proc_r_seqs.append(r_seq[i:i + max_seq_len])
                    proc_pid_seqs.append(pid_seq[i:i + max_seq_len])
                    proc_diff_seqs.append(diff_seq[i:i + max_seq_len])

                    i += max_seq_len

                proc_q_seqs.append(
                    np.concatenate(
                        [
                            q_seq[i:],
                            np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                        ]
                    )
                )
                proc_r_seqs.append(
                    np.concatenate(
                        [
                            r_seq[i:],
                            np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                        ]
                    )
                )
                proc_pid_seqs.append(
                    np.concatenate(
                        [
                            pid_seq[i:],
                            np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                        ]
                    )
                )
                proc_diff_seqs.append(
                    np.concatenate(
                        [
                            diff_seq[i:],
                            np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                        ]
                    )
                )
            
            # test set 의 경우 max_length를 기준으로 시퀀스를 자르고 길이 넘어가는 앞에 부분은 버림
            else : 
                if len(q_seq) > max_seq_len : 
                    proc_q_seqs.append(q_seq[len(q_seq) - max_seq_len:])
                    proc_r_seqs.append(r_seq[len(q_seq) - max_seq_len:])
                    proc_pid_seqs.append(pid_seq[len(q_seq) - max_seq_len:])
                    proc_diff_seqs.append(diff_seq[len(q_seq) - max_seq_len:])

                else : 
                    proc_q_seqs.append(
                        np.concatenate(
                            [
                                q_seq[i:],
                                np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                            ]
                        )
                    )
                    proc_r_seqs.append(
                        np.concatenate(
                            [
                                r_seq[i:],
                                np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                            ]
                        )
                    )
                    proc_pid_seqs.append(
                        np.concatenate(
                            [
                                pid_seq[i:],
                                np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                            ]
                        )
                    )
                    proc_diff_seqs.append(
                        np.concatenate(
                            [
                                diff_seq[i:],
                                np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                            ]
                        )
                    )
                
        return proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_diff_seqs, train_add, test_add