import os

import torch
import numpy as np
import pickle
from sklearn.model_selection import KFold, train_test_split

from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds

class StandardScaler:
    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def build(self, train_data):
        self.train_mean = train_data.mean()
        self.train_std = train_data.std()

    def normalize(self, df):
        return (df - self.train_mean) / self.train_std
        

def main(args):

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.max_seq_len = 30 if args.group_mode == 'userid_with_testid' else args.max_seq_len

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.train_file_name)
    preprocess.load_test_data(args.test_file_name)
    with open(os.path.join(args.asset_dir, "preprocess.pckl"), "wb") as f:
        pickle.dump(preprocess.duration_normalizer, f)

    train_data = preprocess.get_train_data()
    test_data = preprocess.get_test_data()

    total_index = np.arange(len(train_data))
    print(f"total index length: {len(total_index)}")
    if args.valid_mode == 'kfold':
        kf = KFold(n_splits = 5, shuffle = True, random_state = args.seed)
        total_auc, total_acc = 0.0, 0.0 
        for idx, (train_idx, valid_idx) in enumerate(kf.split(total_index)):
            trainset, validset = preprocess.split_data(train_data, test_data, train_idx, valid_idx)
            print(f"[SHAPE CHECK] trainset: {len(trainset)}, validset: {len(validset)}")

            model = trainer.get_model(args).to(args.device)
            best_auc, best_acc = trainer.run(args, trainset, validset, model, idx + 1)
            total_auc += best_auc
            total_acc += best_acc
        print(f"[KFOLD TOTAL SCORE] MEAN AUC: {total_auc / 5}, MEAN ACC: {total_acc / 5}")
        
    elif args.valid_mode == 'random':
        train_idx, valid_idx = train_test_split(total_index, test_size = 0.2, shuffle = True, random_state = args.seed)
        trainset, valid_data = train_data[train_idx], train_data[valid_idx]

        if args.group_mode == 'userid_with_testid':
            valid_for_train, validset = preprocess.concat_for_train(valid_data, True)
            trainset = np.append(trainset, valid_for_train)
        else:
            validset = valid_data
        
        print(f"[SHAPE CHECK] trainset: {len(trainset)}, validset: {len(validset)}")
        model = trainer.get_model(args).to(args.device)
        best_auc, best_acc = trainer.run(args, trainset, validset, model, 0)
        print(f"[SCORE] AUC: {best_auc}, ACC: {best_acc}")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
