import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, ModifiedTransformer
from .optimizer import get_optimizer
from .scheduler import get_scheduler


def run(args, train_data, valid_data, model, fold_num):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args, fold_num
        )

        ### VALID
        auc, acc = validate(valid_loader, model, args, fold_num)

        if auc > best_auc:
            best_auc = auc
            best_acc = acc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                fold_num,
                "model.pt",
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}, \n\n"
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)

    return best_auc, best_acc

def train(train_loader, model, optimizer, scheduler, args, fold_num):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    tk0 = tqdm.tqdm(train_loader, desc = "TRAINING", smoothing=0, mininterval=1.0)
    for step, batch in enumerate(tk0):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        preds = model(input)
        targets = input[-3]  # correct

        loss = compute_loss(preds, targets, args)
        update_params(loss, model, optimizer, scheduler, args)

        # if step % args.log_steps == 0:
        #     print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"[FOLD - {fold_num}] TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args, fold_num):
    model.eval()

    total_preds = []
    total_targets = []
    tk0 = tqdm.tqdm(valid_loader, desc = "VALIDATION", smoothing=0, mininterval=1.0)
    for step, batch in enumerate(valid_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)
        targets = input[-3]  # correct

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"[FOLD - {fold_num}] VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(args, test_data, model):

    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)
        preds = F.sigmoid(preds)
        # predictions
        preds = preds[:, -1]
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    return total_preds


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "lstmattn":
        model = LSTMATTN(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == 'modifiedtf':
        model = ModifiedTransformer(args)

    return model


# 배치 전처리
def process_batch(batch):
    # print(f"[BATCH]:\n {batch}")
    test, question, tag, duration, assess_ratio, correct, mask = batch

    # change to float
    mask = mask.float()
    correct = correct.float()

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).int()
    question = ((question + 1) * mask).int()
    tag = ((tag + 1) * mask).int()

    # print(f"[CORRECT IN PROCESS BATCH]: \n {correct}")
    return (test, question, tag, duration, assess_ratio, correct, mask, interaction)


# loss계산하고 parameter update!
def compute_loss(preds, targets, args):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets.float())

    # 마지막 시퀀드에 대한 값만 loss 계산
    if args.computing_loss == 'last':
        loss = loss[:, -1]
        loss = torch.mean(loss)
    elif args.computing_loss == 'all':
        loss = torch.sum(loss, dim = -1)
        loss = torch.mean(loss)
    elif args.computing_loss == 'custom':
        loss_1 = torch.sum(loss[:, :-1], dim = -1)
        loss = 0.5 * torch.mean(loss_1) + 0.5 * torch.mean(loss[:, -1])
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, fold_num, model_filename):
    print("saving model ...")
    ppath = Path(os.path.join(model_dir, f"fold_{fold_num}", model_filename))
    ppath.parent.mkdir(parents = True, exist_ok = True)
    torch.save(state, str(ppath))


def load_model(args, fold_num):

    # model_path = os.path.join(args.model_dir, args.model_name)
    model_path = os.path.join(args.model_dir, f"fold_{fold_num}", args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
