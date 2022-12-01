import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):

        test, question, tag, _, mask, interaction, duration = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)


        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                duration,
            ],
            2,
        )

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

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

        test, question, tag, _, mask, interaction = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

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
        test, question, tag, _, mask, interaction = input
        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

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
        self.hidden_dim = self.args.hidden_dim
        self.n_heads = self.args.n_heads

        self.embedding_interaction = nn.Embedding(3, self.hidden_dim)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim)
        

        self.duration_emb = nn.Linear(1, self.hidden_dim, bias=False)
        
        self.multi_en = nn.MultiheadAttention( embed_dim= self.hidden_dim, num_heads= self.n_heads, dropout=0.1  )     # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block( self.hidden_dim)                                            # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm( (self.hidden_dim))
        self.layer_norm2 = nn.LayerNorm( (self.hidden_dim))

        self.use_lstm = self.args.use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim , num_layers=1)

        self.out = nn.Linear(in_features= self.hidden_dim , out_features=1)

    
    def forward(self, input):
    # def forward(self, in_ex, in_cat, in_in, first_block=True):
        first_block = True
        count = 0
        test, question, tag, _, mask, interaction, duration = input
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        if first_block:
            print("start", count)

            # 신나는 embedding

            embed_interaction = self.embedding_interaction(interaction)
            embed_interaction = nn.Dropout(0.1)(embed_interaction)

            embed_test = self.embedding_test(test)
            embed_test = nn.Dropout(0.1)(embed_test)

            embed_question = self.embedding_question(question)
            embed_question = nn.Dropout(0.1)(embed_question)

            embed_tag = self.embedding_tag(tag)
            embed_tag = nn.Dropout(0.1)(embed_tag)

            #in_pos = self.embd_pos( in_pos )
            #combining the embedings
            duration = torch.log(duration+1)
            duration = duration.view(-1, 1) # [batch*seq_len, 1]
            duration = self.duration_emb(duration) # [batch*seq_len, d_model]
            embed_duration = duration.view(-1, seq_len, self.hidden_dim) # [batch, seq_len, d_model]


            out = embed_interaction + embed_test + embed_question + embed_tag + embed_duration #+ in_pos                      # (b,sequence,d) [64,100,128]
            print("out shape : ",out.shape)
            count += 1
        else:
            print("not")
            out = embed_interaction
        
        #in_pos = get_pos(self.seq_len)
        #in_pos = self.embd_pos( in_pos )
        #out = out + in_pos                                      # Applying positional embedding

        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape ) [100, 64, 128]
        print("change out ", out.shape) 
        #Multihead attention                            
        n,_,_ = out.shape #100, sequence_len


        out = self.layer_norm1(out)                           # Layer norm
        skip_out = out 
        
        # print("out -1 ", out[-1:,:,:], out[-1:,:,:].shape)
        out, attn_wt = self.multi_en( out[-1:,:,:] , out , out)         # Q,K,V
        #                        #attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        #print('MLH out shape', out.shape)
        out = out + skip_out                                    # skip connection (residual)

        #LSTM
        print("lstm전")
        if self.use_lstm:
            out,_ = self.lstm( out )                                  # seq_len, batch, input_size
            # out = out[-1:,:,:]

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        print("last change:", out.shape)
        out = self.layer_norm2( out )                           # Layer norm 
        skip_out = out
        print("layernorm2")
        out = self.ffn_en( out )
        print("ffn_en")
        out = out + skip_out                                    # skip connection
        print(out.shape)
        print("batchsize", batch_size)
        # out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        # print(out.shape)

        out = self.out( out ).view(batch_size, -1)
        print("end")
        print(out.shape)
        
        return out

# def get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# def get_mask(seq_len):
#     ##todo add this to device
#     return torch.from_numpy( np.triu(np.ones((1 ,seq_len)), k=1).astype('bool'))

# def get_pos(seq_len):
#     # use sine positional embeddinds
#     return torch.arange( seq_len ).unsqueeze(0) 