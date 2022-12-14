import os
import wandb

import torch
import numpy as np
import pickle
from sklearn.model_selection import KFold, train_test_split

from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds


def main(args):
    wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.train_file_name)
    preprocess.load_test_data(args.test_file_name)
    with open(os.path.join(args.asset_dir, "preprocess.pckl"), "wb") as f:
        pickle.dump(preprocess.duration_normalizer, f)

    train_data = preprocess.get_train_data()
    test_data = preprocess.get_test_data()

    wandb.init(project="gru-lastqt", entity="recommendu", config=vars(args))
    wandb.run.name = "model: {0} hdim: {1} nlayer: {2} lr: {3}, head: {4}, len :{5} batch :{6} c_loss:{7} valid:{8} window {9} strid {10}".format(wandb.config["model"],wandb.config['hidden_dim'], wandb.config['n_layers'], wandb.config['lr'], wandb.config['n_heads'], wandb.config['max_seq_len'],wandb.config['batch_size'],wandb.config["computing_loss"],wandb.config["valid_mode"], wandb.config["window"], wandb.config["stride"])
    print(wandb.run.name) 



    total_index = np.arange(len(train_data))
    print(f"total index length: {len(total_index)}")
    if args.valid_mode == 'kfold':
        kf = KFold(n_splits = 5, shuffle = True, random_state = args.seed)
        total_auc, total_acc = 0.0, 0.0 
        for idx, (train_idx, valid_idx) in enumerate(kf.split(total_index)):
            trainset, validset = preprocess.split_data(train_data, test_data, train_idx, valid_idx)
            print(f"[SHAPE CHECK] trainset: {len(trainset)}, validset: {len(validset)}")

            model = trainer.get_model(args).to(args.device)
            print(model)
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
        print(model)
        best_auc, best_acc = trainer.run(args, trainset, validset, model, 0)
        print(f"[SCORE] AUC: {best_auc}, ACC: {best_acc}")
    
    



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
