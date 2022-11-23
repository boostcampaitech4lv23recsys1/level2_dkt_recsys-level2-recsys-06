import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier

from preprocessor import Preprocessor
from utils import split_data

class Trainer():
    def __init__(self, args, params):
        self.preprocessor = Preprocessor(args)
        self.args = args
        self.sub_data_path = os.path.join(self.args.data_path, "sample_submission.csv")
        self.sub_df = pd.read_csv(self.sub_data_path, low_memory = False)\
                        .sort_values(by = ['id']).reset_index(drop = True)
        self.params = params
        self.probs = np.zeros((5, 744))

        self.sum_auc = 0.0
        self.sum_acc = 0.0

    def train_boosting_model(self, trainset, validset, testset, categ_features):
        boost_cl = CatBoostClassifier(**self.params,  random_state = 42, verbose = False)
        boost_cl.fit(trainset.drop(['answerCode'], axis = 1), trainset[['answerCode']], 
                cat_features = categ_features, early_stopping_rounds = 10)

        y_hat = boost_cl.predict(validset.drop(['answerCode'], axis = 1))
        prob = boost_cl.predict_proba(validset.drop(['answerCode'], axis = 1))
        prob_test = boost_cl.predict_proba(testset.drop(['answerCode'], axis = 1))

        auc_score = roc_auc_score(validset['answerCode'], prob[:,1])
        acc_score = accuracy_score(validset['answerCode'], y_hat)
        print(f"AUC score: {auc_score}, Acc score: {acc_score}")
        self.sum_auc += auc_score
        self.sum_acc += acc_score
        
        return prob_test

    def training(self):
        print("Preprocessing training data...")
        train_df, test_df, categ_features = self.preprocessor.preprocess_dataset()
        df_for_split = train_df[train_df['userID'] != train_df['userID'].shift(-1)][['userID', 'answerCode']]

        if self.args.valid_mode == 'kfold':
            kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
            for idx, (train_index, valid_index) in enumerate(kf.split(df_for_split)):
                print(f"[FOLD: {idx + 1}] catboost")
                trainset, validset, testset = split_data(train_df, test_df, train_index, valid_index)
                prob = self.train_boosting_model(trainset, validset, testset, categ_features)
                self.probs[idx] = prob[:, 1]
            print(f"[AVERAGE SCORE] - mean auc: {self.sum_auc / 5}, mean acc: {self.sum_acc / 5}")
            prob = self.fold_ensemble()
        
        else:
            train_index, valid_index = train_test_split(df_for_split, test_size = 0.2, shuffle = True, random_state = 42)
            train_index = list(train_index['userID'])
            valid_index = list(valid_index['userID'])

            trainset, validset, testset = split_data(train_df, test_df, train_index, valid_index)
            prob = self.train_boosting_model(trainset, validset, testset, categ_features)
            prob = prob[:, 1]
            print(f"[SCORE] - valid auc: {self.sum_auc}, acc: {self.sum_acc}")
        
        self.sub_df['prediction'] = prob
        return self.sub_df
        

    def fold_ensemble(self):
        return np.mean(self.probs, axis = 0)