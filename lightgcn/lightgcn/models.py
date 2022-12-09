import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN

from typing import Optional, Union
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
import torch.nn as nn
from torch.nn import Embedding, ModuleList

from torch_geometric.nn.conv import LGConv

EPS = 1e-10
class OrderedLightGCN(LightGCN):
    def __init__(
        self,
        order_alpha: float,
        n_user: int,
        n_item: int,
        num_nodes: int, embedding_dim: int, num_layers: int, alpha: Optional[Union[float, Tensor]] = None, **kwargs,
    ):
        super().__init__(num_nodes, embedding_dim, num_layers, alpha, **kwargs)
        self.order_alpha = order_alpha
        self.user_embedding = nn.Embedding(n_user, embedding_dim)
        self.item_embedding = nn.Embedding(n_item, embedding_dim)
        self.user_order_embedding = nn.Embedding(n_user, embedding_dim)
        self.item_order_embedding = nn.Embedding(n_item, embedding_dim)
        self.clamping()
    
    def clamping(self) :
        self.user_order_embedding.weight.data = torch.clamp(self.user_order_embedding.weight.data, min=EPS)
        self.item_order_embedding.weight.data = torch.clamp(self.item_order_embedding.weight.data, min=EPS)
    
    def get_embedding(self, edge_index: Adj) -> Tensor:

        self.ordered_user_embedding = self.user_embedding.weight + self.user_order_embedding.weight
        self.ordered_item_embedding = self.item_embedding.weight + self.item_order_embedding.weight

        self.order_embedding = torch.cat([
            self.ordered_user_embedding,
            self.ordered_item_embedding
            ],dim= 0)

        # x = self.embedding.weight
        x = self.order_embedding
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out

    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (Tensor): The predictions.
            edge_label (Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_pos, loss_neg = 0,0

        # positive sample에 대한 Loss 계산
        pos_idx = (edge_label == 1).nonzero(as_tuple=False)
        loss_pos = self.item_order_embedding(pos_idx) - self.user_order_embedding(pos_idx)
        loss_pos[loss_pos < EPS] = EPS
        loss_pos = torch.sum(torch.sum(loss_pos ** 2, 1))

        # negative sample에 대한 Loss 계산
        neg_idx = (edge_label == 0).nonzero(as_tuple=False)
        loss_neg = self.item_order_embedding(neg_idx) - self.user_order_embedding(neg_idx)
        loss_neg[loss_neg < EPS] = EPS
        loss_neg = torch.sum(loss_neg ** 2, 1)
        loss_neg = self.order_alpha - loss_neg
        loss_neg[loss_neg < EPS] = EPS
        loss_neg = torch.sum(loss_neg)

        order_loss = loss_pos + loss_neg
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return order_loss + loss_fn(pred, edge_label.to(pred.dtype))

def build(order_alpha, n_user, n_item, n_node, weight=None, logger=None, **kwargs):
    model = OrderedLightGCN(order_alpha, n_user, n_item, n_node, **kwargs)
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def train(
    model,
    train_data,
    valid_data=None,
    n_epoch=100,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:500000]
        # print(eids) -> [      0       1       2 ... 2475959 2475960 2475961] : 총 interaction 개수 =  2475962
        # breakpoint()
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epoch):
        # forward
        pred = model(train_data["edge"])
        loss = model.link_pred_loss(pred, train_data["label"])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prob = model.predict_link(valid_data["edge"], prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"].to("cpu").detach().numpy(), prob > 0.5)
            auc = roc_auc_score(valid_data["label"].to("cpu").detach().numpy(), prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb

                wandb.log(dict(loss=loss, acc=acc, auc=auc))

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Best AUC"
                )
                best_auc, best_epoch = auc, e
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )
                if use_wandb:
                    wandb.log(dict(best_auc=best_auc, best_epoch=best_epoch))

    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred
