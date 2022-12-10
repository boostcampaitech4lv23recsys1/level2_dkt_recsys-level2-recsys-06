import math
import os
from datetime import datetime

import torch
import wandb
import numpy as np
import gc

from .criterion import get_criterion
from .dataloader import get_loaders
from .metric import get_metric
from .models.saintplus import SaintPlus
from .optimizer import get_optimizer
from .scheduler import get_scheduler


def run(args, train_data, valid_data, model):
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    model.to(args.device)

    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )

    # warmup_steps란 무엇일까?
    args.warmup_steps = args.total_steps // 10
    
    loss_fn = torch.nn.BCELoss()
    loss_fn.to(args.device)

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    train_losses = []
    val_losses = []
    val_aucs = []
    metrics = [train_losses, val_losses, val_aucs]
    best_auc = -1
    early_stopping_counter = 0

    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args, loss_fn, metrics
        )

        ### VALID
        auc, acc = validate(valid_loader, model, args, loss_fn, metrics)

        ## TODO: model save or early stopping
        wandb.log(
            {
                "epoch": epoch,
                "train_loss_epoch": train_loss,
                "train_auc_epoch": train_auc,
                "train_acc_epoch": train_acc,
                "valid_auc_epoch": auc,
                "valid_acc_epoch": acc,
            }
        )
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                args,
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                "model.pt",
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)
    

def train(train_loader, model, optimizer, scheduler, args, loss_fn, metrics):

    model.train()
    [train_losses, _, _] = metrics

    total_preds = []
    total_targets = []
    losses = []

    # train_loader의 구조가 어떻게 되어 있나?
    for step, batch in enumerate(train_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        preds = model(input)
        targets = input[-1]  # 정답여부
        interactions = input[-2]

        loss_mask = (interactions != 0)
        preds_masked = torch.masked_select(preds, loss_mask)
        label_masked = torch.masked_select(targets, loss_mask)
        loss = loss_fn(preds_masked, label_masked)

        # loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # # predictions
        # preds = preds[:, -1]
        # targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss.item())

    total_preds = torch.cat(total_preds).detach().cpu().numpy()
    total_targets = torch.cat(total_targets).detach().cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)

    train_losses.append(loss_avg)


    print(f"TRAIN AUC : {auc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args, loss_fn, metrics):

    model.eval()
    [_, val_losses, val_aucs] = metrics

    total_preds = []
    total_targets = []
    losses = []

    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            input = list(map(lambda t: t.to(args.device), process_batch(batch)))
            preds = model(input)
            targets = input[-1]  # correct
            interactions = input[-2]

            loss_mask = (interactions != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(targets, loss_mask)
            loss = loss_fn(preds_masked, label_masked)
            # predictions
            # preds = preds[:, -1]
            # targets = targets[:, -1]

            total_preds.append(preds.detach())
            total_targets.append(targets.detach())
            losses.append(loss)

    total_preds = torch.cat(total_preds).detach().cpu().numpy()
    total_targets = torch.cat(total_targets).detach().cpu().numpy()
    loss_avg = sum(losses) / len(losses)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    val_losses.append(loss_avg)
    val_aucs.append(auc)
    print(f"VALID AUC : {auc}\n")

    return auc, acc


def inference(args, test_data, model):
    
    print("=========================Start inference=========================")
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        # test_input
        preds = model(input)

        # predictions
        preds = preds[:, -1]
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)
    
    #test_total_preds
    now = datetime.now()
    write_path = os.path.join(args.output_dir, f"submission_{now}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))

    print("=========================result saved=========================")


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    # if args.model == "lstm":
    #     model = LSTM(args)
    # if args.model == "lstmattn":
    #     model = LSTMATTN(args)
    # if args.model == "bert":
    #     model = Bert(args)
    
    if args.model == 'saintplus':
        model = SaintPlus(args)
        
    return model


# 배치 전처리
def process_batch(batch):
    # batch는 [colnumber+target+mask, batch_size, seq_len]의 형태
    question, tag, test, test_front, item_mean, item_sum, test_mean, test_sum, assessmentItemIDElo, userIDElo, userID, user_acc, user_total_answer, test_frontElo, month, day, hour, duration, correct, mask = batch
    # test, question, tag, correct, mask = batch

    # change to float
    mask = mask.float()
    correct = correct.float()

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다. (패딩값이 0, 실제로 의미있는 값들은 1과 2)
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    # 카테고리컬 변수들
    test = ((test + 1) * mask).int()
    question = ((question + 1) * mask).int()
    tag = ((tag + 1) * mask).int()
    test_front = ((test_front + 1) * mask).int()
    userID = ((userID + 1) * mask).int()
    month = ((month + 1) * mask).int()
    day = ((day + 1) * mask).int()
    hour = ((hour + 1) * mask).int()

    # 수치형 변수들도 마스킹을 해줘야하나? 아닐거같은데
    # item_mean, item_sum, test_mean, test_sum, assessmentItemIDElo, userIDElo, userID, user_acc, user_total_answer, test_frontElo, month, day, hour, duration, interaction

    # 리턴해줄 때는 마지막 칼럼으로 타겟값을 배치해줌
    return (question, tag, test, test_front, item_mean, item_sum, test_mean, test_sum, assessmentItemIDElo, userIDElo, userID, user_acc, user_total_answer, test_frontElo, month, day, hour, duration, mask, interaction, correct)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(args, state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, args.model_name))


def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
