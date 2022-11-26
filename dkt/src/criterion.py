import torch.nn as nn


def get_criterion(pred, target):
    loss = nn.BCEWithLogitsLoss(reduction="mean")
    return loss(pred, target)
