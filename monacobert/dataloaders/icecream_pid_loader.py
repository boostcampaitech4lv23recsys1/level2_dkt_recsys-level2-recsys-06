import numpy as np
import pandas as pd

from torch.utils.data import Dataset

DATASET_DIR = "../../data/preprocessed_df.csv"

class ICECREAM_PID(Dataset):
    def __init__(self, max_seq_len, config=None, dataset_dir=DATASET_DIR) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir

        # 추가
        self.config = config
        self.train_usernum = 6698   # Hard Coding
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, \
            self.r_list, self.q2idx, self.u2idx, self.pid_seqs, self.pid_list = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]

        self.q_seqs, self.r_seqs, self.pid_seqs, train_add, test_add = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, max_seq_len)

        self.len = len(self.q_seqs)
        self.train_len = self.train_usernum + train_add 

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.pid_seqs[index]

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

        u_idx = np.arange(int(len(u_list)))
        u_train_idx = u_idx[ :self.train_usernum]

        train_u_idx = u_train_idx[ : int(len(u_list) * self.config.train_ratio) ]
        valid_u_idx = u_train_idx[ int(len(u_list) * self.config.train_ratio) : ]
        test_idx = u_idx[self.train_usernum: ]

        q_seqs = []
        r_seqs = []
        pid_seqs = []

        for idx, u in enumerate(u_list):
            df_u = df[df["user_id"] == u]

            q_seq = np.array([q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correct"].values
            pid_seq = np.array([pid2idx[pid] for pid in df_u["item_id"].values])

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            pid_seqs.append(pid_seq)

        return q_seqs, r_seqs, q_list, u_list, r_list, q2idx, u2idx, pid_seqs, pid_list

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []

        train_add = 0
        test_add = 0
        
        for idx, (q_seq, r_seq, pid_seq) in enumerate(zip(q_seqs, r_seqs, pid_seqs)):

            i = 0
            # while i + max_seq_len < len(q_seq): 
            #     if idx < self.train_usernum : train_add += 1
            #     else : test_add += 1
            #     proc_q_seqs.append(q_seq[i:i + max_seq_len])
            #     proc_r_seqs.append(r_seq[i:i + max_seq_len])
            #     proc_pid_seqs.append(pid_seq[i:i + max_seq_len])

            #     i += max_seq_len

            # train set 의 경우 max_length를 기준으로 시퀀스를 잘라서 모두 넣어줌
            if idx < self.train_usernum : 
                while i + max_seq_len < len(q_seq): 
                    train_add += 1
                    proc_q_seqs.append(q_seq[i:i + max_seq_len])
                    proc_r_seqs.append(r_seq[i:i + max_seq_len])
                    proc_pid_seqs.append(pid_seq[i:i + max_seq_len])

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
            
            # test set 의 경우 max_length를 기준으로 시퀀스를 자르고 길이 넘어가는 앞에 부분은 버림
            else : 
                if len(q_seq) > max_seq_len : 
                    proc_q_seqs.append(q_seq[len(q_seq) - max_seq_len:])
                    proc_r_seqs.append(r_seq[len(q_seq) - max_seq_len:])
                    proc_pid_seqs.append(pid_seq[len(q_seq) - max_seq_len:])

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

        return proc_q_seqs, proc_r_seqs, proc_pid_seqs, train_add, test_add