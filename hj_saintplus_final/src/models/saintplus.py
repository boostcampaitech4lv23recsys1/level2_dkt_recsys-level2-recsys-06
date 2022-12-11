from tqdm import notebook
from collections import OrderedDict

import time
import datetime
from datetime import datetime

import pandas as pd
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import math
import warnings
warnings.filterwarnings("ignore")


class FFN(nn.Module):
    def __init__(self, d_ffn, d_model, dropout):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn) #[batch, seq_len, ffn_dim]
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(d_ffn, d_model) #[batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        return self.dropout(x)


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=1000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.scale = nn.Parameter(torch.ones(1))

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(
#             0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.scale * self.pe[:x.size(0), :]
#         return self.dropout(x)


class SaintPlus(nn.Module):
    
    def __init__(self, args):
        super(SaintPlus, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.dropout = self.args.dropout
        self.d_ffn = 2*self.hidden_dim
       
        # 이것까지 추가해줘야 SaintPlus
        self.FFN = FFN(self.d_ffn, self.hidden_dim, dropout=self.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.pos_emb = nn.Embedding(self.args.max_seq_len, self.hidden_dim)

        ### Embedding 
        # ENCODER embedding
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim)
        self.embedding_test_front = nn.Embedding(self.args.n_tag + 1, self.hidden_dim)
        self.embedding_userid = nn.Embedding(self.args.n_userid + 1, self.hidden_dim)
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim)
        self.embedding_day = nn.Embedding(self.args.n_day + 1, self.hidden_dim)
        self.embedding_hour = nn.Embedding(self.args.n_hour + 1, self.hidden_dim)
        
        
        self.embedding_item_mean = nn.Linear(1, self.hidden_dim, bias=False)
        self.embedding_item_sum = nn.Linear(1, self.hidden_dim, bias=False)
        self.embedding_test_mean = nn.Linear(1, self.hidden_dim, bias=False)
        self.embedding_test_sum = nn.Linear(1, self.hidden_dim, bias=False)
        # self.embedding_assessmentItemElo = nn.Linear(1, self.hidden_dim, bias=False)
        # self.embedding_userIDElo = nn.Linear(1, self.hidden_dim, bias=False)
        self.embedding_user_acc = nn.Linear(1, self.hidden_dim, bias=False)
        self.embedding_user_total_answer = nn.Linear(1, self.hidden_dim, bias=False)
        # self.embedding_test_frontElo = nn.Linear(1, self.hidden_dim, bias=False)
        self.embedding_duration = nn.Linear(1, self.hidden_dim, bias=False)


        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim)*8, self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim)*10, self.hidden_dim)

        # # Positional encoding
        # self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        # self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        # self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf')).type(torch.float32)

    def forward(self, input):
        question, tag, test, test_front, item_mean, item_sum, test_mean, test_sum, assessmentItemIDElo, userIDElo, userID, user_acc, user_total_answer, test_frontElo, month, day, hour, duration, mask, interaction, _ = input

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)
    

        # 신나는 embedding
        # ENCODER
        embed_test = self.embedding_test(test.long())
        embed_question = self.embedding_question(question.long())
        embed_tag = self.embedding_tag(tag.long())
        embed_test_front = self.embedding_test_front(test_front.long())
        embed_userid = self.embedding_userid(userID.long())
        embed_month = self.embedding_month(month.long())
        embed_day = self.embedding_day(day.long())
        embed_hour = self.embedding_hour(hour.long())
        
        # embed_item_mean
        embed_item_mean = torch.log(item_mean+1)
        embed_item_mean = embed_item_mean.view(-1, 1) # (batch*seq_len, 1)
        embed_item_mean = self.embedding_item_mean(embed_item_mean) # (batch*seq_len, hidden_dims)
        embed_item_mean = embed_item_mean.view(-1, seq_len, self.hidden_dim) # (batch, seq_len, hidden_dims)
        
        #embed_item_sum
        embed_item_sum = torch.log(item_sum+1)
        embed_item_sum = embed_item_sum.view(-1, 1) 
        embed_item_sum = self.embedding_item_sum(embed_item_sum) 
        embed_item_sum = embed_item_sum.view(-1, seq_len, self.hidden_dim) 

        #embed_test_mean
        embed_test_mean = torch.log(test_mean+1)
        embed_test_mean = embed_test_mean.view(-1, 1) 
        embed_test_mean = self.embedding_item_sum(embed_test_mean) 
        embed_test_mean = embed_test_mean.view(-1, seq_len, self.hidden_dim) 

        # embed_test_sum
        embed_test_sum = torch.log(test_sum+1)
        embed_test_sum = embed_test_sum.view(-1, 1) 
        embed_test_sum = self.embedding_test_sum(embed_test_sum) 
        embed_test_sum = embed_test_sum.view(-1, seq_len, self.hidden_dim) 


        # embed_question_Elo = torch.log(assessmentItemIDElo+1)
        # embed_question_Elo = embed_question_Elo.view(-1, 1) 
        # embed_question_Elo = self.embedding_assessmentItemElo(embed_question_Elo) 
        # embed_question_Elo = embed_question_Elo.view(-1, seq_len, self.hidden_dim) 


        # embed_user_Elo = torch.log(userIDElo+1)
        # embed_user_Elo = embed_user_Elo.view(-1, 1) 
        # embed_user_Elo = self.embedding_userIDElo(embed_user_Elo) 
        # embed_user_Elo = embed_user_Elo.view(-1, seq_len, self.hidden_dim) 


        embed_user_acc = torch.log(user_acc+1)
        embed_user_acc = embed_user_acc.view(-1, 1) 
        embed_user_acc = self.embedding_user_acc(embed_user_acc) 
        embed_user_acc = embed_user_acc.view(-1, seq_len, self.hidden_dim) 


        embed_user_total = torch.log(user_total_answer+1)
        embed_user_total = embed_user_total.view(-1, 1) 
        embed_user_total = self.embedding_user_total_answer(embed_user_total) 
        embed_user_total = embed_user_total.view(-1, seq_len, self.hidden_dim) 


        # embed_test_elo = torch.log(test_frontElo+1)
        # embed_test_elo = embed_test_elo.view(-1, 1) 
        # embed_test_elo = self.embedding_test_frontElo(embed_test_elo) 
        # embed_test_elo = embed_test_elo.view(-1, seq_len, self.hidden_dim) 


        embed_duration = torch.log(duration+1)
        embed_duration = embed_duration.view(-1, 1) 
        embed_duration = self.embedding_duration(embed_duration) 
        embed_duration = embed_duration.view(-1, seq_len, self.hidden_dim) 

        embed_enc = torch.cat([embed_test,
                               embed_question,
                               embed_tag,
                               embed_test_front,
                               embed_item_mean,
                               embed_item_sum,
                               embed_test_mean,
                               embed_test_sum,
                            #    embed_question_Elo,
                            #    embed_test_elo,
                               ], dim=-1)
        embed_enc = self.enc_comb_proj(embed_enc)

        # embed_enc = embed_test + embed_question + embed_tag

        embed_interaction = self.embedding_interaction(interaction.long())

        embed_dec = torch.cat([
            # embed_test,
                               embed_question,
                               embed_tag,
                               embed_userid,
                            #    embed_user_Elo,
                               embed_user_acc,
                               embed_user_total,
                               embed_month,
                               embed_day,
                               embed_hour,
                               embed_duration,
                               embed_interaction], dim=-1)

        embed_dec = self.dec_comb_proj(embed_dec)
        
        # embed_dec = embed_duration + embed_interaction

        embed_enc = embed_enc.type(torch.float32)
        embed_dec = embed_dec.type(torch.float32)
        
        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        # if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
        #     self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        # if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
        #     self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        # if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
        #     self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
       
        
        # 만약 잘 안되면 이거 꼭 하게~~~~~
        pos = torch.arange(seq_len).unsqueeze(0).to(self.device)
        pos_emb = self.pos_emb(pos)

        embed_enc += pos_emb
        embed_dec += pos_emb

        over_head_mask = self.get_mask(seq_len).to(self.device)
        
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        # embed_enc = self.pos_encoder(embed_enc)
        # embed_dec = self.pos_decoder(embed_dec)
        

        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=over_head_mask,
                               tgt_mask=over_head_mask,
                               memory_mask=over_head_mask)
                               
        out = self.layer_norm(out)
        out = out.permute(1, 0, 2)
        # out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        final_out = self.FFN(out)
        final_out = self.layer_norm(final_out+out)
        final_out = self.fc(out)

        preds = torch.sigmoid(final_out).squeeze(-1)
        return preds

