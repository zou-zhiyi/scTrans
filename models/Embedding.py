import torch
import torch.nn as nn


class WordEmbedding(nn.Module):
    def __init__(self, d_model, vocab, dropout_rate):
        super(WordEmbedding, self).__init__()
        self.word_embed = nn.Embedding(vocab, d_model, padding_idx=0)
        # self.dropout = nn.Dropout1d(dropout_rate)

    def forward(self, x):
        # return self.word_embed(x)
        # print(f'word embedding input shape:{x.shape}')
        return self.word_embed(x)


class PercentValueEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        # d_model=512,dropout=0.1
        super(PercentValueEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Dropout1d(dropout)

    def forward(self, feature, feature_value=None, embedding_dropout=False):
        if feature_value is None:
            return feature

        # pe = torch.zeros(feature.size()[0], feature.size()[1], self.d_model).to(feature.device).requires_grad_(False)
        # div_term = torch.ones(self.d_model).to(pe.device)
        # print(f'feature shape: {feature.shape}')
        # print(f'feature_value shape: {feature_value.shape}')
        # print(f'result shape: {feature.mul(feature_value).shape}')
        # pe[:, :, :] = feature_value * div_term
        # print(f'mul shape: {feature.mul(feature_value).shape}')
        # return self.dropout(feature)
        # print(f'embedding dropout: {embedding_dropout}')
        if embedding_dropout is True:
            # print('chek embedding dropout')
            return self.dropout(feature.mul(feature_value))
        else:
            return feature.mul(feature_value)


class MlpPositionValueEncoding(nn.Module):
    def __init__(self, d_model, dropout, position_dim=4):
        # d_model=512,dropout=0.1
        super(MlpPositionValueEncoding, self).__init__()
        self.linear = nn.Linear(d_model + position_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_dim = position_dim
        # self.dropout = nn.Dropout1d(dropout)

    def value_encoding(self, feature, feature_value):
        # print(f'before feature value shape:{feature_value.shape}')
        feature_value = feature_value.repeat(1, 1, self.position_dim)
        # print(f'after feature value shape:{feature_value.shape}')
        # print(f'before feature value shape:{feature.shape}')
        x = torch.cat((feature, feature_value), dim=-1)
        x = self.linear(x)
        return x

    def forward(self, feature, feature_value=None, embedding_dropout=False):
        if feature_value is None:
            return feature

        if embedding_dropout is True:
            return self.dropout(self.value_encoding(feature, feature_value))
        else:
            return self.value_encoding(feature, feature_value)


class AbsPositionValueEncoding(nn.Module):
    def __init__(self, d_model, dropout):
        # d_model=512,dropout=0.1
        super(AbsPositionValueEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, feature, feature_value=None):
        if feature_value is None:
            return self.dropout(feature)
        # print(feature.shape)
        batch, word_num = feature.shape[0], feature.shape[1]
        pe = torch.zeros((batch, word_num, self.d_model)).to(feature.device)
        x = torch.pow(10000,
                      torch.arange(0, self.d_model, 2, dtype=torch.float32, device=feature.device) / self.d_model)
        # print(pe.shape)
        # print(feature_value.shape)
        # print(x.shape)
        # print(f'pe shape :{pe[:, :, 0::2].shape}')
        # print(f' feature_value shape: {feature_value.shape}')
        pe[:, :, 0::2] = torch.sin((feature_value / x).view(batch, word_num, -1))
        pe[:, :, 1::2] = torch.cos((feature_value / x).view(batch, word_num, -1))
        feature = feature + pe
        return self.dropout(feature)


class LearnableAbsolutePositionEmbedding(nn.Module):
    def __init__(self, max_position_embeddings, dmodel):
        super().__init__()
        self.embedding = nn.Embedding(max_position_embeddings, dmodel)

    def forward(self, feature, feature_val):
        if feature_val is None:
            return feature
        # if feature_val.dim() != 3:
        #     feature_val = feature_val.unsqueeze(-1)
        feature_val = feature_val.long()
        feature_val = feature_val.squeeze(-1)
        feature_val = self.embedding(feature_val)
        # print(feature.shape)
        # print(feature_val.shape)
        return feature + feature_val


class ModelPercentEmbedding(nn.Module):
    def __init__(self, d_model, dropout_rate, vocab=30000):
        super(ModelPercentEmbedding, self).__init__()
        self.feature_embedding = WordEmbedding(d_model, vocab, dropout_rate)
        self.value_embedding = PercentValueEncoding(d_model, dropout_rate)

    def forward(self, feature_idx, feature_val=None, embedding_dropout=False):
        # print(feature_idx)
        x = self.feature_embedding(feature_idx)
        # print(self.feature_embedding.word_embed.weight.grad)
        return self.value_embedding(x, feature_val, embedding_dropout)

class ModelMlpPositionEmbedding(nn.Module):
    def __init__(self, d_model, dropout_rate, vocab=30000):
        super(ModelMlpPositionEmbedding, self).__init__()
        self.feature_embedding = WordEmbedding(d_model, vocab, dropout_rate)
        self.value_embedding = MlpPositionValueEncoding(d_model, dropout_rate)

    def forward(self, feature_idx, feature_val=None, embedding_dropout=False):
        # print(feature_idx)
        x = self.feature_embedding(feature_idx)
        # print(self.feature_embedding.word_embed.weight.grad)
        return self.value_embedding(x, feature_val, embedding_dropout)
class ModelAbsPositionEmbedding(nn.Module):

    def __init__(self, d_model, dropout_rate, vocab=30000):
        super(ModelAbsPositionEmbedding, self).__init__()
        self.feature_embedding = WordEmbedding(d_model, vocab, dropout_rate)
        self.value_embedding = AbsPositionValueEncoding(d_model, dropout_rate)

    def forward(self, feature_idx, feature_val=None):
        x = self.feature_embedding(feature_idx)
        return self.value_embedding(x, feature_val)


class ModelLearnableAbsolutePositionEmbedding(nn.Module):

    def __init__(self, d_model, dropout_rate, vocab=30000):
        super(ModelLearnableAbsolutePositionEmbedding, self).__init__()
        self.feature_embedding = WordEmbedding(d_model, vocab, dropout_rate)
        self.value_embedding = LearnableAbsolutePositionEmbedding(max_position_embeddings=10, dmodel=d_model)

    def forward(self, feature_idx, feature_val=None):
        x = self.feature_embedding(feature_idx)
        return self.value_embedding(x, feature_val)
