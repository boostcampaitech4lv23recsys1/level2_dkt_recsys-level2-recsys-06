
import numpy as np

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
import lightgbm as lgb



class LightGBM:

    def __init__(self, args, data):
        super().__init__()

        self.train_data = data['train_data']
        self.test_data = data['test_data']
        self.features = data['features']
        self.device = args.DEVICE
        self.cv = args.CV
        
        self.random_state = args.SEED
        
        self.verbose_eval = args.VERBOSE_EVAL
        self.num_boost_round = args.NUM_BOOST_ROUND
        self.early_stopping_rounds = args.EARLY_STOPPING_ROUND

        if self.cv == 'kfold':
            self.n_splits = args.N_SPLITS
            self.shuffle = args.DATA_SHUFFLE
            self.predicts = np.zeros((self.n_splits, self.test_data.shape[0]))
        
        else:
            pass

        
    def train(self): # 훈련  # 우선  kfold만 넣어 놓음
        kf = KFold(n_splits = self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        df_for_split = self.train_data[self.train_data['userID'] != self.train_data['userID'].shift(-1)][['userID', 'answerCode']]
        sum_auc, sum_acc = 0, 0
        

        for idx, (train_index, valid_index) in enumerate(kf.split(df_for_split)):
                print(f"Cross Validation training starts: {idx+1} of {self.n_splits}")
                train_data = self.train_data[self.train_data['userID'].isin(train_index)]
                valid_data = self.train_data[self.train_data['userID'].isin(valid_index)]
                
                y_train = train_data['answerCode']
                train = train_data.drop(['answerCode'], axis=1)
                
                y_valid = valid_data['answerCode']
                valid = valid_data.drop(['answerCode'], axis=1)
                
                lgb_train = lgb.Dataset(train[self.features], y_train)
                lgb_valid = lgb.Dataset(valid[self.features], y_valid)
                
                model = lgb.train(
                {'objective': 'binary'}, 
                lgb_train,
                valid_sets=[lgb_train, lgb_valid],
                verbose_eval=self.verbose_eval,
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds,
                )

                valid_pred = model.predict(valid[self.features])
                auc_score = roc_auc_score(y_valid, valid_pred)
                acc_score = accuracy_score(y_valid, np.where(valid_pred >= 0.5, 1, 0))
                
                test_preds_lg = model.predict(self.test_data[self.features])
                self.predicts[idx] = test_preds_lg

                print(f"[LGBM FOLD: {idx + 1}] AUC score: {auc_score}, Acc score: {acc_score}")
                sum_auc += auc_score
                sum_acc += acc_score
                print('\n\n')

        print(f"[LGBM AVERAGE SCORE] - mean auc: {sum_auc / 5}, mean acc: {sum_acc / 5}")


        return np.mean(self.predicts, axis=0)



class CatBoost:
    def __init__(self, args, data):
        super().__init__()

        self.train_data = data['train_data']
        self.test_data = data['test_data']
        self.features = data['features']
        self.random_state = args.SEED
        self.cv = args.CV

        self.device = args.DEVICE
        self.verbose = args.VERBOSE
        

        if self.cv == 'kfold':
            self.n_splits = args.N_SPLITS
            self.shuffle = args.DATA_SHUFFLE
            self.predicts = np.zeros((self.n_splits, self.test_data.shape[0]))
        
        else:
            pass        
        
        return


    def train(self):
        kf = KFold(n_splits = self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
        categorical_features = ['month', 'test_front', 'KnowledgeTag', 'userID', 'assessmentItemID']

        # 이후 args로 고쳐보기
        params = {'iterations':100, 'learning_rate':0.3, 'depth':10, 'eval_metric':'AUC'}

        df_for_split = self.train_data[self.train_data['userID'] != self.train_data['userID'].shift(-1)][['userID', 'answerCode']]
        sum_auc, sum_acc = 0, 0

        for idx, (train_index, valid_index) in enumerate(kf.split(df_for_split)):

            train_data = self.train_data[self.train_data['userID'].isin(train_index)]
            valid_data = self.train_data[self.train_data['userID'].isin(valid_index)]
            
            y_train = train_data['answerCode']
            train = train_data.drop(['answerCode'], axis=1)
            
            y_valid = valid_data['answerCode']
            valid = valid_data.drop(['answerCode'], axis=1)
                
            cboost_cl = CatBoostClassifier(**params, random_state=self.random_state, verbose = self.verbose)
            cboost_cl.fit(train[self.features], y_train, cat_features = categorical_features)  # , early_stopping_rounds = 10

            valid_pred = cboost_cl.predict_proba(valid[self.features])[:,1]
            auc_score = roc_auc_score(y_valid, valid_pred)
            acc_score = accuracy_score(y_valid, np.where(valid_pred >= 0.5, 1, 0))
            
            test_preds_cb = cboost_cl.predict_proba(self.test_data[self.features])[:,1]
            self.predicts[idx] = test_preds_cb
            
            print(f"[CatBoost FOLD: {idx + 1}] AUC score: {auc_score}, Acc score: {acc_score}")
            sum_auc += auc_score
            sum_acc += acc_score
            print('\n\n')        

        print(f"[CatBoost AVERAGE SCORE] - mean auc: {sum_auc / 5}, mean acc: {sum_acc / 5}")


        return np.mean(self.predicts, axis=0)


    # def predict_Train(self):
    #     return
    # def predict(self, dataloader):
    #     return
