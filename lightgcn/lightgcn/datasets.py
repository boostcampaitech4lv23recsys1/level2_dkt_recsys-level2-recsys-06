import os
from config import CFG

import pandas as pd
import torch


def prepare_dataset(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    train_data, valid_data, test_data = separate_data(data)
    id2index = indexing_data(data)
    train_data_proc = process_data(train_data, id2index, device)
    valid_data_proc = process_data(valid_data, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)

    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(valid_data, "Valid", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)

    return train_data_proc, valid_data_proc, test_data_proc, len(id2index)


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2], ignore_index=True)
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data):
    test_data = data[data.answerCode < 0]
    test_idx = test_data.index
    train_data = data[data.answerCode >= 0]
    
    # test set에 있는 모든 유저의 sequence의 CFG.valid_num개의 문제 풀이 데이터를 valid data로 추가
    valid_idx = []
    if CFG.user_wandb:
        import wandb
        for i in range(1,wandb.config['valid_num'] + 1) :
            tmp_idx = list(map(lambda x:x -i, test_idx))
            valid_idx += tmp_idx
    else :
        for i in range(1,CFG.valid_num + 1) :
            tmp_idx = list(map(lambda x:x -i, test_idx))
            valid_idx += tmp_idx
        
    # train set에 있는 모든 유저의 sequence의 마지막 문제 풀이 데이터를 valid data로 추가
    valid_idx += list(train_data.index.to_series().groupby(train_data['userID']).last().reset_index(name='last_idx')['last_idx'].values)
    valid_data = data.loc[valid_idx]    

    train_data = train_data.drop(index=valid_idx)

    return train_data, valid_data, test_data


def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
