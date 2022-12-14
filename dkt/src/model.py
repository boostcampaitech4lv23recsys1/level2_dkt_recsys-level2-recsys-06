import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from .layer import SASRecBlock
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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

class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.embed_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.embed_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.embed_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.embed_dim)
        self.embedding_cont = nn.Linear(2, self.embed_dim, bias = False)
        # embedding combination projection
        self.comb_proj = nn.Linear(self.embed_dim * (4 + 1), self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):

        # test, question, tag, _, mask, interaction = input
        test, question, tag, duration, assess_ratio, _ , mask, interaction = input

        """
        print(f"[TEST]:\n {test[0]}")
        print(f"[QUESTION]:\n {question[0]}")
        print(f"[TAG]:\n {tag[0]}")
        print(f"[DURATION]:\n {duration[0]}")
        print(f"[MASK]:\n {mask[0]}")
        print(f"[INTERACTION]:\n {interaction[0]}\n\n")
        """

        # print(f"duration shape: {duration.shape}, assess_ratio shape: {assess_ratio.shape}")
        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        x_cont = torch.cat([duration.unsqueeze(2), assess_ratio.unsqueeze(2)], dim = -1)
        embed_cont = self.embedding_cont(x_cont)
        embed = torch.cat([
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_cont], dim = -1)

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.embed_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.embed_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.embed_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.embed_dim)
        self.embedding_cont = nn.Linear(2, self.embed_dim, bias = False)
        # embedding combination projection
        self.comb_proj = nn.Linear(self.embed_dim * (4 + 1), self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):

        test, question, tag, duration, assess_ratio,lastid, correct, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        x_cont = torch.cat([duration.unsqueeze(2), assess_ratio.unsqueeze(2)], dim = -1)
        embed_cont = self.embedding_cont(x_cont)
        embed = torch.cat([
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_cont], dim = -1)

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]
        # print(f"encoded_layers: \n  \n {encoded_layers}, \n\n")
        # print(f"encoded_layers: \n {sequence_output.shape}  \n {sequence_output}, \n\n")

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.embed_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.embed_dim)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.embed_dim)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.embed_dim)
        self.embedding_cont = nn.Linear(2, self.embed_dim, bias = False)
        # embedding combination projection
        self.comb_proj = nn.Linear(self.embed_dim * (4 + 1), self.hidden_dim)

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def forward(self, input):
        test, question, tag, duration, assess_ratio, _ , mask, interaction = input
        # print('\n', mask.shape, '\n', mask)
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)
        x_cont = torch.cat([duration.unsqueeze(2), assess_ratio.unsqueeze(2)], dim = -1)
        embed_cont = self.embedding_cont(x_cont)
        embed = torch.cat([
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_cont], dim = -1)

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out

"""
Encoder --> GRU --> dense

"""

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )


class last_query_model(nn.Module):
    """
    Embedding --> MLH --> GRU
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_len = self.args.max_seq_len
        self.hidden_dim = self.args.hidden_dim // 4
        self.n_heads = self.args.n_heads

        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim)
        self.embedding_lastid = nn.Embedding(self.args.n_lastid +1, self.hidden_dim)

        self.comb_proj = nn.Linear((self.hidden_dim) * 6, self.hidden_dim)
        

        self.duration_emb = nn.Linear(1, self.hidden_dim, bias=False)
        self.ratio_emb = nn.Linear(1, self.hidden_dim, bias=False)
        
        self.multi_en = nn.MultiheadAttention( embed_dim= self.hidden_dim, num_heads= self.n_heads, dropout=0.1)    
        self.ffn_en = Feed_Forward_block( self.hidden_dim)                                          
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)


        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        # self.use_lstm = self.args.use_lstm
        # if self.use_lstm:
        #     self.lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim , num_layers=1)
        self.gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        self.out = nn.Linear(in_features= self.hidden_dim , out_features=1)
        self.activation = nn.Sigmoid()
        

    
    def forward(self, input):
        first_block = True
        count = 0

        test, question, tag, duration, assess_ratio,lastid, correct, mask, interaction = input
        
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)


        embed_interaction = self.embedding_interaction(interaction)
        # embed_interaction = nn.Dropout(0.1)(embed_interaction)

        embed_test = self.embedding_test(test)
        # embed_test = nn.Dropout(0.1)(embed_test)

        embed_question = self.embedding_question(question)
        # embed_question = nn.Dropout(0.1)(embed_question)

        embed_tag = self.embedding_tag(tag)
        # embed_tag = nn.Dropout(0.1)(embed_tag)

        # embed_lastid = self.embedding_lastid(lastid)
        # embed_lastid = nn.Dropout(0.1)(embed_lastid)

        #in_pos = self.embd_pos( in_pos )
        #combining the embedings
        duration = torch.log(duration+1)
        duration = duration.view(-1, 1) # [batch*seq_len, 1]
        duration = self.duration_emb(duration) # [batch*seq_len, d_model]
        embed_duration = duration.view(-1, seq_len, self.hidden_dim) # [batch, seq_len, d_model]

        assess_ratio = torch.log(assess_ratio+1)
        assess_ratio = assess_ratio.view(-1, 1) # [batch*seq_len, 1]
        assess_ratio = self.ratio_emb(assess_ratio) # [batch*seq_len, d_model]
        embed_ratio = assess_ratio.view(-1, seq_len, self.hidden_dim) # [batch, seq_len, d_model]

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_duration,
                embed_ratio
            ],
            2,
        )
        embed = self.comb_proj(embed)

        # out = embed_interaction + embed_test + embed_question + embed_tag + embed_duration + embed_ratio#+ in_pos                      # (b,sequence,d) [64,100,128]
        
        
        #in_pos = get_pos(self.seq_len)
        #in_pos = self.embd_pos( in_pos )
        #out = out + in_pos                                      # Applying positional embedding

        
        q = self.query(embed).permute(1, 0, 2)

        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)

        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        out, _ = self.multi_en(q, k, v)

        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.layer_norm1(out)

        ## feed forward network
        out = self.ffn_en(out)

        ## residual + layer norm
        out = embed + out
        out = self.layer_norm2(out)

        # out, hidden = self.lstm(out)
        out, hidden = self.gru(out)

        
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.out(out)
        out = out.view(batch_size, -1)
        
        return out

class gru_lastquery(nn.Module):
    """
    Embedding --> GRU --> MLH
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_len = self.args.max_seq_len
        self.hidden_dim = self.args.hidden_dim // 4
        self.n_heads = self.args.n_heads

        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim)
        self.embedding_lastid = nn.Embedding(self.args.n_lastid +1, self.hidden_dim)

        self.comb_proj = nn.Linear((self.hidden_dim) * 6, self.hidden_dim)
        

        self.duration_emb = nn.Linear(1, self.hidden_dim, bias=False)
        self.ratio_emb = nn.Linear(1, self.hidden_dim, bias=False)
        
        self.multi_en = nn.MultiheadAttention( embed_dim= self.hidden_dim, num_heads= self.n_heads, dropout=0.1  )     # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block( self.hidden_dim)                                            # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)


        self.query = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim
        )

        self.gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        self.out = nn.Linear(in_features= self.hidden_dim , out_features=1)
        self.activation = nn.Sigmoid()
        

    
    def forward(self, input):
        test, question, tag, duration, assess_ratio,lastid, correct, mask, interaction = input
        
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        embed_interaction = self.embedding_interaction(interaction)
        # embed_interaction = nn.Dropout(0.1)(embed_interaction)

        embed_test = self.embedding_test(test)
        # embed_test = nn.Dropout(0.1)(embed_test)

        embed_question = self.embedding_question(question)
        # embed_question = nn.Dropout(0.1)(embed_question)

        embed_tag = self.embedding_tag(tag)
        # embed_tag = nn.Dropout(0.1)(embed_tag)

        # embed_lastid = self.embedding_lastid(lastid)
        # embed_lastid = nn.Dropout(0.1)(embed_lastid)

        #in_pos = self.embd_pos( in_pos )
        #combining the embedings
        duration = torch.log(duration+1)
        duration = duration.view(-1, 1) # [batch*seq_len, 1]
        duration = self.duration_emb(duration) # [batch*seq_len, d_model]
        embed_duration = duration.view(-1, seq_len, self.hidden_dim) # [batch, seq_len, d_model]

        assess_ratio = torch.log(assess_ratio+1)
        assess_ratio = assess_ratio.view(-1, 1) # [batch*seq_len, 1]
        assess_ratio = self.ratio_emb(assess_ratio) # [batch*seq_len, d_model]
        embed_ratio = assess_ratio.view(-1, seq_len, self.hidden_dim) # [batch, seq_len, d_model]

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_duration,
                embed_ratio
            ],
            2,
        )
        embed = self.comb_proj(embed)

        # out = embed_interaction + embed_test + embed_question + embed_tag + embed_duration + embed_ratio#+ in_pos                      # (b,sequence,d) [64,100,128]

        #in_pos = get_pos(self.seq_len)
        #in_pos = self.embd_pos( in_pos )
        #out = out + in_pos                                      # Applying positional embedding

        out, hidden = self.gru(embed)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        q = self.query(out).permute(1, 0, 2)

        q = self.query(out)[:, -1:, :].permute(1, 0, 2)

        k = self.key(out).permute(1, 0, 2)
        v = self.value(out).permute(1, 0, 2)

        ## attention
        # last query only
        out1, _ = self.multi_en(q, k, v)

        ## residual + layer norm
        out1 = out1.permute(1, 0, 2)
        out1 = out1 + out
        out1 = self.layer_norm1(out1)

        ## feed forward network
        out1 = self.ffn_en(out1)

        ## residual + layer norm
        out1 = out1 + out
        out1 = self.layer_norm2(out1)

        
        out1 = out1.contiguous().view(batch_size, -1, self.hidden_dim)
        out1 = self.out(out1)
        out1 = out1.view(batch_size, -1)

        
        return out1
