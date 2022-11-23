import time
import pandas as pd
import os
import random
import numpy as np
import torch 
import torch.nn as nn



from data_process import data_processing, context_data_split, context_data_loader, stratified_kfold
from train import FactorizationMachineModel

train = pd.read_csv('~/input/data/train_data.csv')
fm_test = pd.read_csv('~/input/data/test_data_no_sub.csv')
test = pd.read_csv('~/input/data/fm_test_data.csv')
sub = pd.read_csv('~/input/data/fm_submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

predicts_list = []
def main():
    seed_everything(9)
    data = data_processing(train,fm_test,test,sub)
    for i in range(5):
        data = stratified_kfold(data,i)
        data = context_data_loader(data)
        model = FactorizationMachineModel(data)

        model.train()
        predicts = model.predict(data['test_dataloader'])
        predicts_list.append(predicts)
        
    sub['answerCode'] = np.mean(predicts_list, axis=0)
    now = time.localtime()
    now_date = time.strftime('%Y%m%d', now)
    now_hour = time.strftime('%X', now)
    save_time = now_date + '_' + now_hour.replace(':', '')
    sub.to_csv('{}_{}.csv'.format(save_time,"FM"), index=False)

main()
