import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from models.Embedding import ModelPercentEmbedding, ModelAbsPositionEmbedding, \
    ModelLearnableAbsolutePositionEmbedding, ModelMlpPositionEmbedding
from models.Encoder import EnhanceEncoder
from torch.autograd import Function


class Encoder(nn.Module):
    def __init__(self, d_model, h_dim, head, d_ffw, dropout_rate, enhance_num, vocab=300000, embedding_dropout=False):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding_dropout = embedding_dropout
        self.embedding = ModelPercentEmbedding(d_model=d_model, vocab=vocab, dropout_rate=dropout_rate)
        self.encoder = EnhanceEncoder(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw, dropout_rate=dropout_rate,
                                      is_attention_dropout=True, enhance_num=enhance_num)
        self.projection_head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(),
                                             nn.Linear(2 * d_model, d_model))

    def forward(self, query_idx, query_val, key_idx, key_val, mask=None, key_pre_attention=None, proj_nums=None):
        query_embedding = self.embedding(query_idx, None, False)
        key_embedding = self.embedding(key_idx, key_val, self.embedding_dropout)
        # key_embedding = self.embedding(key_idx, key_val, False)
        encode, attn_score = self.encoder(query_embedding, key_embedding, key_embedding, key_pre_attention, mask)
        return encode, attn_score

    def encode_with_projection_head(self, query_idx, query_val, key_idx, key_val, mask=None, key_pre_attention=None):
        encode, attn_score = self.forward(query_idx, query_val, key_idx, key_val, mask, key_pre_attention)
        return self.projection_head(encode), encode, attn_score


class ClassificationModel(nn.Module):
    def __init__(self, model: nn.Module, d_model, predict_type, dropout_rate=0, mlp_layer=None):
        super(ClassificationModel, self).__init__()
        if mlp_layer is None:
            mlp_layer = []
        self.encode = model
        self.mlp_linear_list = nn.ModuleList([])
        self.bn_list = nn.ModuleList([])
        self.mlp_size = len(mlp_layer)
        pre = d_model
        for i in range(0, self.mlp_size):
            self.mlp_linear_list.append(nn.Linear(pre, mlp_layer[i]))
            self.bn_list.append(nn.BatchNorm1d(mlp_layer[i]))
            pre = mlp_layer[i]
        self.predict_linear = nn.Linear(pre, predict_type)
        self.dropout = nn.Dropout(dropout_rate)
        # self.my_layerNorm = MyLayerNorm()
        self.a = nn.ReLU()

    def freeze_core(self):
        self.encode.requires_grad_(requires_grad=False)

    def load_core(self, core_state):
        self.encode.load_state_dict(core_state)

    def forward(self, query_idx, query_val, key_idx, key_val, mask=None, key_pre_attention=None, proj_nums=None):
        # print(proj_num)
        x, _ = self.encode(query_idx, query_val, key_idx, key_val, mask, key_pre_attention)
        x = x.squeeze(-2)
        # print(f'x shape: {x.shape}')
        for i in range(self.mlp_size):
            # x = self.a(self.mlp_linear_list[i](x))
            x = self.a(self.bn_list[i](self.mlp_linear_list[i](x)))
            x = self.dropout(x)
        x = self.predict_linear(x)
        # x = self.a(x)
        # x = self.a(self.predict_linear(x))
        return x


class EnhancedClassificationModel(ClassificationModel):

    def forward(self, query_idx, query_val, key_idx, key_val, mask=None):
        # key_val = self.my_layerNorm(key_idx, key_val)
        x, _ = self.encode(query_idx, query_val, key_idx, key_val, mask=mask)

        # print(f'x shape: {x.shape}')
        for i in range(self.mlp_size):
            x = self.a(self.mlp_linear_list[i](x))
            x = self.dropout(x)
        x = self.predict_linear(x)
        # x = self.a(x)
        # x = self.a(self.predict_linear(x))
        return x


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0, eps=1e-8):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.eps = eps

    def forward(self, x, target):
        # prd_num = x.shape[-1]
        # x = x[target < prd_num]
        # target = target[target< prd_num]
        # target = target.unsqueeze(-1)
        logprobs = torch.softmax(x, dim=-1)

        # print(f'logprobs: {logprobs.shape}')
        # print(f'target: {target.unsqueeze(-1).shape}')
        # print(f'percent bigger 0.9 num: {torch.nonzero(logprobs > 0.9)}')
        logprobs = torch.log(logprobs + self.eps)
        nll_loss = -logprobs.gather(dim=-1, index=target)
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, t=0.5):
        super(ContrastiveLoss, self).__init__()
        self.t = t

    def forward(self, encode_1, encode_2, label_mask=None):
        batch_size = encode_1.shape[0]
        v_len = encode_1.shape[-1]
        t = self.t * math.sqrt(v_len)
        encode = torch.cat([encode_1, encode_2], dim=0)
        if torch.any(torch.isinf(encode)):
            print('encode inf')
        if torch.any(torch.isnan(encode)):
            print('encode nan')
        encode = encode.squeeze(1)
        encode = encode / torch.norm(encode, dim=-1, keepdim=True)
        encode_sim = torch.mm(encode, encode.t().contiguous())
        sim_matrix = torch.exp(encode_sim / t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=encode.device)).bool()
        sim_matrix_new = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_mask = torch.cat([torch.eye(batch_size, device=encode.device),
                              torch.eye(batch_size, device=encode.device)], dim=0)
        pos_mask = torch.cat([pos_mask, pos_mask], dim=-1)
        pos_mask = (pos_mask - torch.eye(2 * batch_size, device=encode.device)).bool()
        if label_mask is not None:
            pos_mask = pos_mask | label_mask
        pos_sim = sim_matrix / (sim_matrix_new.sum(-1) + 1e-7)
        if torch.count_nonzero(pos_sim) != pos_sim.numel():
            print("log error")
        if torch.any(torch.isnan(pos_sim)):
            print('pos nan')

        pos_sim = -torch.log(pos_sim + 1e-7)
        pos_sim = pos_sim * pos_mask
        # print(torch.max(pos_sim, -1)[0])
        # print(pos_sim)
        # tmp = pos_sim.detach().cpu().numpy()
        # print(pos_sim.sum(dim=-1))
        # print(torch.count_nonzero(pos_mask, dim=-1))
        # print(pos_sim.sum(dim=-1) / torch.count_nonzero(pos_mask, dim=-1))
        # print(f'pos sim shape: {pos_sim.shape}')
        # print(pos_sim.sum())
        # print(torch.count_nonzero(pos_mask))
        return pos_sim.sum() / torch.count_nonzero(pos_mask)



def generate_enhance_core_model(d_model, h_dim, head, d_ffw, dropout_rate, vocab, enhance_num,
                                device_name="cpu", mlp_layer=None):
    device = torch.device(device_name)
    model_core = Encoder(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw,
                         dropout_rate=dropout_rate,
                         vocab=vocab,
                         enhance_num=enhance_num)
    model_core.to(device)
    return model_core

def generate_enhance_classification_model_with_d(d_model, h_dim, head, d_ffw, dropout_rate, predict_type, vocab,
                                                 enhance_num,
                                                 freeze=False,
                                                 device_name="cpu", mlp_layer=None):
    device = torch.device(device_name)
    model_core = Encoder(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw,
                         dropout_rate=dropout_rate,
                         vocab=vocab,
                         enhance_num=enhance_num)
    model_core.to(device)
    model = ClassificationModel(model=model_core, d_model=d_model, predict_type=predict_type, dropout_rate=dropout_rate,
                                mlp_layer=mlp_layer)
    model.to(device)
    if freeze:
        model.freeze_core()
    return model

def embedding_pca_initializing(model, adata_list, pca_num, word_idx_idc, device_name, vocab):
    if pca_num is None:
        return
    pca = PCA(n_components=pca_num)
    new_embedding = np.zeros((vocab, pca_num))
    for adata in adata_list:
        gene_vec = pca.fit_transform(adata.X.T)
        print(gene_vec.shape)
        gene_names = adata.var_names
        gene_idx = gene_names.map(lambda x: word_idx_idc.getIdx(x.lower()))
        gene_idx = np.array(gene_idx).astype(int)
        print(f'gene idx: {gene_idx}')
        print(f'gene idx: {gene_idx[gene_idx >= 0]}')
        new_embedding[gene_idx[gene_idx >= 0]] = gene_vec[gene_idx >= 0]
    model.embedding.feature_embedding.word_embed = \
        torch.nn.Embedding.from_pretrained(torch.tensor(new_embedding, dtype=torch.float32,
                                                        device=torch.device(device_name)), freeze=False, padding_idx=0)
    print('using pca embedding')
    print(model.embedding.feature_embedding.word_embed.weight.data)


def embedding_pca_initializing_from_pk(model, adata_list, pca_num, word_idx_idc, device_name, vocab):
    if pca_num is None:
        return
    pca = PCA(n_components=pca_num)
    new_embedding = np.zeros((vocab, pca_num))
    for adata in adata_list:
        gene_vec = pca.fit_transform(adata.X.T)
        print(gene_vec.shape)
        gene_names = adata.var_names
        print(f'gene names: {gene_names}')
        # print(gene_names)
        # check_result = list(map(lambda x: word_idx_idc.getIdx(int(x)) if word_idx_idc.getIdx(int(x)) else x, gene_names))
        # print(f'check_result:{check_result}')
        # gene_idx = gene_names.map(lambda x: word_idx_idc.getIdx(int(x)) if word_idx_idc.getIdx(int(x)) else -1)
        gene_idx = gene_names.map(lambda x: word_idx_idc.getIdx(x.lower()) if word_idx_idc.getIdx(x.lower()) else \
            (word_idx_idc.getIdx(int(x)) if word_idx_idc.getIdx(int(x)) else -1))
        # gene_idx = gene_names.map(lambda x: word_idx_idc.getIdx(x.lower()) if word_idx_idc.getIdx(x.lower()) else -1)

        # gene_idx = gene_names.map(lambda x: word_idx_idc.getIdx(x.lower()))
        # new_embedding = np.random.random((30000, pca_num))
        # print(gene_idx)
        # gene_idx = gene_idx.as_type(int)
        print(f'gene idx: {gene_idx}')
        print(f'gene idx: {gene_idx[gene_idx >= 0]}')
        gene_idx = np.array(gene_idx).astype(int)

        new_embedding[gene_idx[gene_idx >= 0]] = gene_vec[gene_idx >= 0]
    model.embedding.feature_embedding.word_embed = \
        torch.nn.Embedding.from_pretrained(torch.tensor(new_embedding, dtype=torch.float32,
                                                        device=torch.device(device_name)), freeze=False, padding_idx=0)
    print('using pca embedding')
    print(model.embedding.feature_embedding.word_embed.weight.data)
