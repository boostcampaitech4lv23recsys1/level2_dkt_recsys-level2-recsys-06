import torch
import torch.nn as nn

from src.embed import DataEmbedding
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

        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out).view(batch_size, -1)
        return out


class ModifiedTransformer(nn.Module):
    def __init__(self, args):
        super(ModifiedTransformer, self).__init__()
        self.args = args
        self.embed_dim = self.args.embed_dim
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        self.enc_embedding = DataEmbedding(c_in = 1, d_model = self.hidden_dim, args = self.args, dropout = self.args.drop_out)
        self.blocks = nn.ModuleList([SASRecBlock(self.args.n_heads, self.hidden_dim, self.args.drop_out) for _ in range(self.n_layers)])

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, input):

        """
        'using_features': ['userID', 'assessmentItemID', 'testId', 'KnowledgeTag', 
        'relative_time_median', 'hour', 'dayofweek', 
        'duration', 'userIDElo', 'assessmentItemIDElo', 'testIdElo', 'KnowledgeTagElo', 
        'past_correct', 'average_correct', 'mean_time', 'answerCode', 'mask']
        """

        batch_size = input['interaction'].size(0)
        mask = input['mask']
        max_len = self.args.max_seq_len

        pad_attn_mask = mask.data.eq(0)
        pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size , max_len, max_len).unsqueeze(1)

        x = input['interaction']
        x_mark_categ = [input['assessmentItemID'], input['testId'], input['KnowledgeTag'], input['relative_time_median'],
                        input['hour'], input['dayofweek']]
        x_mark_cont = [input['duration'], input['userIDElo'], input['assessmentItemIDElo'], 
                        input['testIdElo'], input['KnowledgeTagElo'], input['past_correct'], 
                        input['average_correct'], input['mean_time']]

        out = self.enc_embedding(x, x_mark_categ, x_mark_cont)

        for block in self.blocks:
            out, attn_dist = block(out, pad_attn_mask)
        out, _ = self.lstm(out)
        # out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out