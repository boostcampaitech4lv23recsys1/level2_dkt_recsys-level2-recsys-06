import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from args import parser
from model import SaintPlus, NoamOpt
from torch.utils.data import DataLoader
from data_generator import Riiid_Sequence, Riiid_Sequence2
import os
import pandas as pd


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    num_layers = args.num_layers
    num_heads = args.num_heads
    d_model = args.d_model
    d_ffn = d_model * 4
    max_len = args.max_len
    n_questions = args.n_questions
    n_parts = args.n_parts
    n_tasks = args.n_tasks
    n_ans = args.n_ans

    seq_len = args.seq_len
    warmup_steps = args.warmup_steps
    dropout = args.dropout
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size

    with open("fianl_sub_val_group.pkl.zip", 'rb') as pick:
        sub_val_group = pickle.load(pick)
    sub_val_seq = Riiid_Sequence2(sub_val_group, args.seq_len)
    sub_val_size = len(sub_val_seq)
    sub_val_loader = DataLoader(sub_val_seq, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = SaintPlus(seq_len=seq_len, num_layers=num_layers, d_ffn=d_ffn, d_model=d_model, num_heads=num_heads,
                      max_len=max_len, n_questions=n_questions, n_tasks=n_tasks, dropout=dropout)

    checkpoint = torch.load('fianlSaint2.pt', map_location=device)
    state_dict = checkpoint['net']
    model.load_state_dict(state_dict)

    model.to(device)

    # model.load_state_dict(torch.load("./saintFinal.pt"))

    model.eval()
    submission = []
    for step, data in enumerate(sub_val_loader):
        content_ids = data[0].to(device).long()
        time_lag = data[1].to(device).float()
        ques_elapsed_time = data[2].to(device).float()
        itemaver = data[3].to(device).float()
        useraver = data[4].to(device).float()
        answer_correct = data[5].to(device).long()
        label = data[6].to(device).float()
        preds = model(content_ids, time_lag, ques_elapsed_time, itemaver, useraver, answer_correct)
        preds1 = preds[:, -1]
        submission.extend(preds1.data.cpu().numpy())

    print(len(submission))
    submission3 = pd.DataFrame()
    submission3['id'] = np.arange(744)
    submission3['prediction'] = submission
    submission3.to_csv("submissionSatin.csv", index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    # os.makedirs(args.model_dir, exist_ok=True)
    main(args)