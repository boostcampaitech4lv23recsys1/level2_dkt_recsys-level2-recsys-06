from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from utils import collate_fn, pid_collate_fn, pid_diff_collate_fn

#icecream
from dataloaders.icecream_pid_diff_loader import ICECREAM_PID_DIFF

def get_loaders(config, idx=None):

    # 1. choose the loaders
    if config.dataset_name == "icecream":
        dataset = ICECREAM_PID_DIFF(config.max_seq_len, config=config)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "ednet_pid":
        dataset = ICECREAM_PID_DIFF(config.max_seq_len, config=config)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "icecream_pid_diff":
        dataset = ICECREAM_PID_DIFF(config.max_seq_len, config=config)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
    else:
        print("Wrong dataset_name was used...")

    # 2. data chunk
    train_size = int( dataset.train_len * config.train_ratio )
    valid_size = dataset.train_len - train_size
    test_size = len(dataset) - dataset.train_len

    train_dataset = Subset(dataset, range( train_size ))
    valid_dataset = Subset(dataset, range( train_size, train_size + valid_size ))
    test_dataset =  Subset(dataset, range( train_size + valid_size, train_size + valid_size + test_size))

    # 3. get DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True, # train_loader use shuffle
        collate_fn = collate
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size,
        shuffle = False, # valid_loader don't use shuffle
        collate_fn = collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = False, # test_loader don't use shuffle
        collate_fn = collate
    )

    return train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff