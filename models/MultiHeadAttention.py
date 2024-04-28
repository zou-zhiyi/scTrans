import torch.nn as nn
import torch
import torch.nn.functional as F
import math


def attention(query, key, value, value_mask=None, mask=None, dropout=None):
    """
    :param value_mask:先验注意力
    :param query: 维度:(batch, head, word_q, head_vec)
    :param key: 维度:(batch, head, word_k, head_vec)
    :param value: d维度:(batch, head, word_k, head_vec)
    :param mask: 维度:(batch, word_k),query和key之间的关系掩码矩阵
    :param dropout: dropout模块
    :return: 注意力机制计算后的结果
    """
    # print(f'query shape: {query.shape}')
    # print(f'key shape: {key.shape}')
    # print(f'value shape: {value.shape}')
    # 转变向量维度，用于计算相似性矩阵,得到矩阵为(batch, head, head_vec, word_k)
    key = torch.transpose(key, -1, -2)
    d_query = query.size(-1)
    # 获取注意力分数，矩阵维度为:(batch, head, word_q, word_k)
    attention_score = torch.matmul(query, key) / math.sqrt(d_query)
    # print(attention_score)
    if value_mask is not None:
        value_mask = value_mask.transpose(-1, -2).unsqueeze(1)
        # print(f'value_mask shape: {value_mask.shape}')
        # 通过缩放分数进行先验注意力
        attention_score = torch.mul(value_mask, attention_score)
    if mask is not None:
        # print(mask.shape)
        mask = mask.unsqueeze(-3)
        # print(mask.shape)
        # mask = mask.unsqueeze(-2)

        attention_score = torch.masked_fill(attention_score, mask == 1, -1e10)
    # softmax放缩注意力分数

    attention_agg = torch.softmax(attention_score, -1)
    attention_score = torch.mean(attention_agg, dim=1)
    if dropout is not None:
        attention_agg = dropout(attention_agg)
    # 聚合后的信息维度:(batch, head, word_q, head_vec)
    attention_agg = torch.matmul(attention_agg, value)
    # print(f"attention agg shape:{attention_agg.shape}")
    # print(f'val enhance agg shape:{torch.matmul(attention_agg, value).shape}')
    # print(attention_score.shape)

    # print(attention_score.shape)
    return attention_agg, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h_dim=None, out_dim=None, head=1, dropout_rate=0.1, is_dropout=False):
        super(MultiHeadAttention, self).__init__()
        if d_model % head != 0:
            raise ValueError("d_model % head != 0, please reset head")
        if h_dim is None:
            h_dim = d_model
        if out_dim is None:
            out_dim = d_model
        self.d_head = h_dim // head
        self.head = head
        self.q = nn.Linear(in_features=d_model, out_features=h_dim)
        self.k = nn.Linear(in_features=d_model, out_features=h_dim)
        self.v = nn.Linear(in_features=d_model, out_features=h_dim)
        self.out = nn.Linear(in_features=h_dim, out_features=out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.is_dropout = is_dropout
        # nn.init.xavier_uniform_(self.q.weight)
        # nn.init.xavier_uniform_(self.k.weight)
        # nn.init.xavier_uniform_(self.v.weight)
        # nn.init.xavier_uniform_(self.out.weight)
        # self.a = nn.Tanh()

    def forward(self, query, key, value, value_mask=None, mask=None):
        """

        :param query: 维度:(batch, word_q, word_vec)
        :param key: 维度:(batch, word_k, word_vec)
        :param value: d维度:(batch, word_k, word_vec)
        :param mask: 维度:(word_q, word_k),query和key之间的关系掩码矩阵
        :return: 注意力聚合后的信息，和各个头的注意力分数
        """
        batch_size = query.size()[0]
        # print(query)
        # print(key)
        # print(value)
        # query, key, value = self.a(self.q(query)), self.a(self.k(key)), self.a(self.v(value))
        # query, key, value = torch.sigmoid(self.q(query)), torch.sigmoid(self.k(key)), torch.sigmoid(self.v(value))
        query, key, value = self.q(query), self.k(key), self.v(value)
        # view操作并转置维度，获取attention计算需要的向量
        query = query.view(batch_size, -1, self.head, self.d_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.head, self.d_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.head, self.d_head).transpose(1, 2)
        if self.is_dropout:
            attention_agg, attention_score = attention(query, key, value, value_mask, mask, self.dropout)
        else:
            attention_agg, attention_score = attention(query, key, value, value_mask, mask)
        # 维度:(batch, head, word_q, head_vec),需要转换成(batch, word_q, word_vec)
        attention_agg = attention_agg.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head * self.d_head)
        return self.out(attention_agg), attention_score
