import numpy as np
import datetime

import torch
from get_modules.get_loaders import get_loaders
from get_modules.get_models import get_models
from get_modules.get_trainers import get_trainers
from utils import get_optimizers, get_crits, recorder, visualizer, setSeeds

from define_argparser import define_argparser

def main(config, train_loader=None, valid_loader=None, test_loader=None, num_q=None, num_r=None, num_pid=None, num_diff=None):
    # 0. device setting
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    # 1. get dataset from loader
    # not use fivefold
    idx = 0
    train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff = get_loaders(config, idx)

    # 2. select models using get_models
    model = get_models(num_q, num_r, num_pid, num_diff, device, config)
    
    # 3. select optimizers using get_optimizers
    optimizer = get_optimizers(model, config)
    
    # 4. select crits using get_crits
    crit = get_crits(config)
    
    # 5. select trainers for models, using get_trainers
    trainer = get_trainers(model, optimizer, device, num_q, crit, config)

    # 6. use trainer.train to train the models
    # the result contain train_scores, valid_scores, hightest_valid_score, highest_test_score
    train_scores, valid_scores, highest_valid_score = trainer.train(train_loader, valid_loader, test_loader, config)

    # 7. model record
    # for model's name
    today = datetime.datetime.today()
    record_time = str(today.month) + "_" + str(today.day) + "_" + str(today.hour) + "_" + str(today.minute)
    # model's path
    model_path = './model_records/' + str(highest_valid_score) + "_" + record_time + "_" + config.model_fn
    # model save
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    return train_scores, valid_scores, highest_valid_score, record_time

# If you used python train.py, then this will be start first
if __name__ == "__main__":
    # get config from define_argparser
    setSeeds()
    config = define_argparser() 

    # if fivefold = False 
    train_auc_scores, valid_auc_scores, best_valid_score, record_time = main(config)
    # # for record
    # recorder(test_auc_score, record_time, config)
    # # for visualizer
    # visualizer(train_auc_scores, valid_auc_scores, record_time)
