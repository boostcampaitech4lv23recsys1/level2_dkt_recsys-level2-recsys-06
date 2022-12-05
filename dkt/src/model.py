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

        test, question, tag, duration, assess_ratio, _ , mask, interaction = input

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
        print(out.shape)
        print("----")
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        print(out.shape)
        print("-----")
        out = self.fc(out).view(batch_size, -1)
        print(out.shape)
        return out

"""
Encoder --> LSTM --> dense

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
    Embedding --> MLH --> LSTM
    """
    def __init__(self, args):
    # def __init__(self , args, dim_model, heads_en, total_ex ,total_cat, total_in,seq_len, use_lstm=True):
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

        self.use_lstm = self.args.use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim , num_layers=1)
        self.gru = nn.GRU(
            self.hidden_dim, self.hidden_dim, self.args.n_layers, batch_first=True
        )

        self.out = nn.Linear(in_features= self.hidden_dim , out_features=1)
        self.activation = nn.Sigmoid()
        

    
    def forward(self, input):
    # def forward(self, in_ex, in_cat, in_in, first_block=True):
        first_block = True
        count = 0
        # test, question, tag, duration, assess_ratio, _ , mask, interaction = input
        test, question, tag, duration, assess_ratio,lastid, correct, mask, interaction = input
        
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        if first_block:
            print("start", count)

            # 신나는 embedding

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
            
            print("out shape : ",embed.shape)
            count += 1
        else:
            print("not")
            embed = embed_interaction
        
        #in_pos = get_pos(self.seq_len)
        #in_pos = self.embd_pos( in_pos )
        #out = out + in_pos                                      # Applying positional embedding

        # out1 = out.permute(1,0,2)
        # out1, _ = self.multi_en(out1[-1:,:,:], out1, out1)
        
        # out1 = out1.permute(1,0,2)
        # out1 = out + out1
        # out1 = self.layer_norm1(out1)
        

        # out1 = self.ffn_en(out1)
        # out1 = out + out1
        # out1 = self.layer_norm2(out1)

        # out1, hidden = self.lstm(out1)
        # out1 = out1.contiguous().view(batch_size, -1, self.hidden_dim)
        # out1 = self.out(out1).view(batch_size, -1)

        ####-------
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
        preds = self.activation(out).view(batch_size, -1)

        
        
        
        
        # ###----
        # out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape ) [100, 64, 128]
        # print("change out ", out.shape) 
        # #Multihead attention                            
        # n,_,_ = out.shape #100, sequence_len


        # out = self.layer_norm1(out)                           # Layer norm
        # skip_out = out 
        
        # # print("out -1 ", out[-1:,:,:], out[-1:,:,:].shape)
        # out, attn_wt = self.multi_en( out[-1:,:,:] , out , out)         # Q,K,V
        # #                        #attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        # #print('MLH out shape', out.shape)
        # out = out + skip_out                                    # skip connection (residual)

   
        # #LSTM
        # print("lstm전")
        # if self.use_lstm:
        #     out,_ = self.lstm( out )                                  # seq_len, batch, input_size
        #     # out = out[-1:,:,:]

        # #feed forward
        # out = out.permute(1,0,2)                                # (b,n,d)
        # print("last change:", out.shape)
        # out = self.layer_norm2( out )                           # Layer norm 
        # skip_out = out
        # print("layernorm2")
        # out = self.ffn_en( out )
        # print("ffn_en")
        # out = out + skip_out                                    # skip connection
        # print(out.shape)
        # print("batchsize", batch_size)
        # # out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # # print(out.shape)

        # out = self.out( out ).view(batch_size, -1)
        # print("end")
        # print(out.shape)

        # #----
        
        return preds

# def get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# def get_mask(seq_len):
#     ##todo add this to device
#     return torch.from_numpy( np.triu(np.ones((1 ,seq_len)), k=1).astype('bool'))

# def get_pos(seq_len):
#     # use sine positional embeddinds
#     return torch.arange( seq_len ).unsqueeze(0) 

class ModifiedTransformer(nn.Module):
    def __init__(self, args):
        super(ModifiedTransformer, self).__init__()
        self.args = args
        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.embed_dim, padding_idx = 0)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.embed_dim, padding_idx = 0)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.embed_dim, padding_idx = 0)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.embed_dim, padding_idx = 0)
        self.embedding_cont = nn.Linear(2, self.embed_dim, bias = False)
        # embedding combination projection
        self.comb_proj = nn.Linear(self.embed_dim * (4 + 1), self.hidden_dim)
        self.position_embedding = PositionalEmbedding(d_model= self.hidden_dim, max_len = self.args.max_seq_len)
        
        self.blocks = nn.ModuleList([SASRecBlock(self.args.n_heads, self.hidden_dim, self.args.drop_out) for _ in range(self.n_layers)])

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):

        # test, question, tag, _, mask, interaction = input
        test, question, tag, duration, assess_ratio, correct , mask, interaction = input


        batch_size = interaction.size(0)
        # mask_pad = torch.BoolTensor(interaction.to('cpu') > 0).unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, max_len)
        # mask_time = (1 - torch.triu(torch.ones((1, 1, interaction.size(1), interaction.size(1))), diagonal=1)).bool() # (batch_size, 1, max_len, max_len)
        # n_mask = (mask_pad & mask_time).to(self.args.device)
        # print(interaction.shape, interaction, '\n')
        # print(n_mask.shape, n_mask, '\n\n')
        max_len = self.args.max_seq_len
        pad_attn_mask = mask.data.eq(0)
        pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size , max_len, max_len).unsqueeze(1)
        # print(pad_attn_mask.shape)
        # print(pad_attn_mask)

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

        out = self.comb_proj(embed)
        
        position_embed = self.position_embedding(out)
        out = out + position_embed

        for block in self.blocks:
            out, attn_dist = block(out, pad_attn_mask)

        # out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out

class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        
        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.args.dim_div)

#         self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
    
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)      

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()
        
        # T-Fixup
        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

        
        
    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*ln.*|.*bn.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param          
            elif re.match(r'encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)
        
    # lastquery는 2D mask가 필요함
    def get_mask(self, seq_len, mask, batch_size):
        new_mask = torch.zeros_like(mask)
        new_mask[mask == 0] = 1
        new_mask[mask != 0] = 0
        mask = new_mask
    
        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, input):
        torch.backends.cudnn.enabled = False
        _, mask, interaction = input[-3:]
        # test, question, tag, duration, assess_ratio, correct , mask, interaction = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_features = []
        for _input, _embedding_feature in zip(input[:-3], self.embedding_features):
            value = _embedding_feature(_input)
            embed_features.append(value)
    
        embed_features = [embed_interaction] + embed_features
        embed = torch.cat(embed_features, 2)

        embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
#         position = self.get_pos(seq_len).to('cuda')
#         embed_pos = self.embedding_position(position)
#         embed = embed + embed_pos

        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        self.mask = self.get_mask(seq_len, mask, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)