import torch
import torch.nn as nn

from models.MultiHeadAttention import MultiHeadAttention
from models.PointWiseFeedForward import PointWiseFeedForward
from models.Sublayer import Sublayer
from models.utils import clones

class EncoderBlock(nn.Module):
    def __init__(self, d_model, h_dim, head, d_ffw, dropout_rate, is_attention_dropout):
        super(EncoderBlock, self).__init__()
        self.attention_model = MultiHeadAttention(d_model=d_model,h_dim=h_dim, out_dim=d_model, head=head, dropout_rate=dropout_rate,
                                                  is_dropout=is_attention_dropout)
        self.point_wise_ffw = PointWiseFeedForward(d_model=d_model, d_ffw=d_ffw, dropout_rate=dropout_rate)
        self.sublayer1 = Sublayer(d_model, dropout_rate)
        self.sublayer2 = Sublayer(d_model, dropout_rate)
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, value_mask=None, mask=None):
        """
        :param value_pre_attention:
        :param value_mask:
        :param query: 输入特征，维度：(batch, word_q, word_vec)
        :param key: 输入特征，维度：(batch, word_k, word_vec)
        :param value: 输入特征，维度：(batch, word_v, word_vec)
        :param mask: 输入特征，维度：(batch, word_q, word_k)
        :return: 编码后的信息
        """
        # attn:输入特征维度：(batch, word_q, word_vec)
        attn, attn_score = self.attention_model(query, key, value, value_mask, mask)
        query = self.sublayer1(attn, query)
        # query = self.sublayer1(attn, None)
        ffw = self.point_wise_ffw(query)
        # print(ffw)
        query = self.sublayer2(ffw, query)
        # query = self.sublayer2(ffw, query)

        # return query, attn_score
        # return attn, attn_score
        # return ffw, attn_score
        return query, attn_score
        # return self.layerNorm(query), attn_score


class EnhanceEncoder(nn.Module):
    def __init__(self, d_model, h_dim, head, d_ffw, dropout_rate, is_attention_dropout, enhance_num=1):
        super().__init__()
        self.encoder_list = clones(EncoderBlock(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw, dropout_rate=dropout_rate,
                                                is_attention_dropout=is_attention_dropout), enhance_num)

    def forward(self, query, key, value, value_mask=None, mask=None):
        # attn:输入特征维度：(batch, word_q, word_vec)
        # print(f'encoder list: {len(self.encoder_list)}')
        for i in range(len(self.encoder_list)):
            query, attn_score = self.encoder_list[i](query, key, value, value_mask, mask)
        return query, attn_score

