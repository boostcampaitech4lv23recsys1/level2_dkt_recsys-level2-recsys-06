import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class CategoricalEmbedding(nn.Module):
    def __init__(self, d_model, args):
        super(CategoricalEmbedding, self).__init__()

        self.assess_embed = nn.Embedding(args.n_questions + 1, d_model, padding_idx = 0)
        self.testid_embed = nn.Embedding(args.n_test + 1, d_model, padding_idx = 0)
        self.knowledge_embed = nn.Embedding(args.n_tag + 1, d_model, padding_idx = 0)
        self.relative_time_embed = nn.Embedding(3, d_model, padding_idx = 0)
        self.hour_embed = nn.Embedding(25, d_model, padding_idx = 0)
        self.dow_embed = nn.Embedding(8, d_model, padding_idx = 0)

    def forward(self, x):
        assess_x = self.assess_embed(x[0])
        testid_x = self.testid_embed(x[1])
        knowledge_x = self.knowledge_embed(x[2])
        relative_time_x = self.relative_time_embed(x[3])
        hour_x = self.hour_embed(x[4])
        dow_x = self.dow_embed(x[5])
        
        return assess_x + testid_x + knowledge_x + relative_time_x + hour_x + dow_x

class NumericalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(NumericalEmbedding, self).__init__()

        self.embed = nn.Linear(8, d_model)
    
    def forward(self, x):
        x = list(map(lambda t: t.unsqueeze(2), x))
        x = torch.cat(x, dim = -1)
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, args, dropout=0.2):
        super(DataEmbedding, self).__init__()

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.categ_embedding = CategoricalEmbedding(d_model = d_model, args = args)
        self.cont_embedding = NumericalEmbedding(d_model = d_model)

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, x_mark_categ, x_mark_cont):
        x = x.unsqueeze(2).float()
        value_embed = self.value_embedding(x)
        position_embed = self.position_embedding(x)
        categ_embed = self.categ_embedding(x_mark_categ)
        cont_embed = self.cont_embedding(x_mark_cont)

        x = value_embed + position_embed + categ_embed + cont_embed
        
        return self.dropout(x)