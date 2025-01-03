import gc

import glob
import os.path
import time
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import sklearn.metrics.pairwise
from sklearn import manifold
from typing import Optional
import matplotlib.pyplot as pl

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from models.dataset import generate_train_test_dataset_list, PadCollate, \
    generate_dataset_list, generate_dataset_list_with_hvg
from models.impl.ContrastiveTrainer import train_enhance_contrastive, train_enhance_contrastive_model
from models.model import LabelSmoothing, generate_enhance_classification_model_with_d
from models.train import Trainer, Context, enhance_classification_construct

from models.utils import set_seed, calculate_score, write_file_to_pickle, \
    check_anndata, write_to_h5ad, read_mapping_file

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, \
    silhouette_score


def feature_val_mean_enhance(feature_val_pad):
    new_feature_val_pad = torch.clone(feature_val_pad)
    new_feature_val_pad[new_feature_val_pad != 0] = 1
    return new_feature_val_pad


def feature_mask_enhance(feature_idx_pad, mask_rate):
    mask = torch.zeros_like(feature_idx_pad, dtype=torch.float32).to(feature_idx_pad.device)
    mask[feature_idx_pad != 0] = mask_rate
    mask = torch.bernoulli(mask).long()
    return feature_idx_pad * mask


class ClassificationTrainer(Trainer):

    def __init__(self, model: nn.Module, train_dataset, test_dataset, device_name='cpu', lr=0.001,
                 weight_decay=1e-2, trained_model_path=None, continue_train=False, save_path=''):
        super().__init__(model, train_dataset, test_dataset, lr=lr, device_name=device_name,
                         weight_decay=weight_decay, trained_model_path=trained_model_path,
                         continue_train=continue_train, save_path=save_path)
        self.threshold = 0.5
        self.cross_entropy_loss = LabelSmoothing(0)

    def train_inner(self, train_loader, context: Optional[Context]):

        loss_sum = []
        y_prd_list = []
        y_true_list = []
        for i, batch in enumerate(train_loader):
            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad, feature_idx_lens, \
                feature_val_lens, label = batch
            current_batch_size = tissue_idx.shape[0]
            if current_batch_size == 1:
                break

            key_padding_mask = torch.zeros_like(feature_idx_pad).to(self.device)
            key_padding_mask[feature_idx_pad == 0] = 1
            key_padding_mask = key_padding_mask.unsqueeze(-2)

            tissue_idx = tissue_idx.unsqueeze(-1)
            tissue_val = tissue_val.unsqueeze(-1).unsqueeze(-1)


            feature_val_pad = feature_val_pad.unsqueeze(-1)

            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad = tissue_idx.to(self.device), \
                tissue_val.to(self.device), feature_idx_pad.to(self.device), feature_val_pad.to(self.device)
            label = label.to(self.device).unsqueeze(-1)

            prd = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad, feature_val_pad,
                             key_padding_mask, None)

            softmax_loss = self.cross_entropy_loss(prd, label)

            self.opt.zero_grad()
            softmax_loss.backward()
            self.opt.step()
            prd_prop = torch.softmax(prd, dim=-1)

            tmp_prd = torch.flatten(torch.argmax(prd_prop, dim=-1)).cpu().numpy()

            y_prd_list = y_prd_list + tmp_prd.tolist()
            y_true_list = y_true_list + torch.flatten(label).cpu().numpy().tolist()


            loss_sum.append(softmax_loss.item())
        train_loss = np.mean(loss_sum)
        context.epoch_loss = train_loss

    def test_inner(self, test_loader, context: Optional[Context]):
        loss_sum = []
        y_prop_list = None
        y_prd_list = []
        y_true_list = []
        for i, batch in enumerate(test_loader):
            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad, feature_idx_lens, \
                feature_val_lens, label = batch


            key_padding_mask = torch.zeros_like(feature_idx_pad).to(self.device)
            key_padding_mask[feature_idx_pad == 0] = 1
            key_padding_mask = key_padding_mask.unsqueeze(-2)
            tissue_idx = tissue_idx.unsqueeze(-1)
            tissue_val = tissue_val.unsqueeze(-1).unsqueeze(-1)


            feature_val_pad = feature_val_pad.unsqueeze(-1)
            label = label.to(self.device)

            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad = tissue_idx.to(self.device), \
                tissue_val.to(self.device), feature_idx_pad.to(self.device), feature_val_pad.to(self.device)

            prd = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad, feature_val_pad,
                             key_padding_mask)

            prd_prop = torch.softmax(prd, dim=-1)

            if y_prop_list is None:
                y_prop_list = prd_prop.detach().cpu().numpy()
            else:
                y_prop_list = np.concatenate((y_prop_list, prd_prop.detach().cpu().numpy()), axis=0)
            pd_max_prop, _ = torch.max(prd_prop, dim=-1)
            tmp_prd = torch.flatten(torch.argmax(prd_prop, dim=-1)).cpu().numpy()

            y_prd_list = y_prd_list + tmp_prd.tolist()
            y_true_list = y_true_list + torch.flatten(label).cpu().numpy().tolist()

            predict_type = context.predict_type
            # 去除未知细胞类型
            prd = prd[label < predict_type, :]
            label = label[label < predict_type]
            label = label.unsqueeze(-1)

            # 基于已知细胞类型计算损失
            softmax_loss = self.cross_entropy_loss(prd, label)

            loss_sum.append(softmax_loss.item())

        context.epoch_loss = np.mean(loss_sum)
        context.cell_type_prd_list = y_prd_list
        context.cell_type_true_list = y_true_list

    def predict_cell_type(self, context: Optional[Context]):
        self.model.eval()
        y_prd_list = []
        n_embedding, n_prop = None, None
        data_loader = context.data_loader
        for i, batch in enumerate(data_loader):
            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad, feature_idx_lens, \
                feature_val_lens = batch

            key_padding_mask = torch.zeros_like(feature_idx_pad).to(self.device)
            key_padding_mask[feature_idx_pad == 0] = 1
            key_padding_mask = key_padding_mask.unsqueeze(-2)
            tissue_idx = tissue_idx.unsqueeze(-1)
            tissue_val = tissue_val.unsqueeze(-1).unsqueeze(-1)

            feature_val_pad = feature_val_pad.unsqueeze(-1)

            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad = tissue_idx.to(self.device), \
                tissue_val.to(self.device), feature_idx_pad.to(self.device), feature_val_pad.to(self.device)

            embedding, _ = self.model.encode(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad,
                                                             feature_val_pad,
                                                             key_padding_mask)
            prd = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad, feature_val_pad,
                             key_padding_mask)

            prd_prop = torch.softmax(prd, dim=-1)

            pd_max_prop, _ = torch.max(prd_prop, dim=-1)
            tmp_prd = torch.flatten(torch.argmax(prd_prop, dim=-1)).cpu().numpy()
            y_prd_list = y_prd_list + tmp_prd.tolist()

            if n_embedding is None:
                n_embedding = embedding.detach().cpu().squeeze(-2).numpy()
                n_prop = prd_prop.detach().cpu().numpy()
                # n_label = tissue_idx.cpu().numpy()
            else:
                n_embedding = np.concatenate((n_embedding, embedding.detach().cpu().squeeze(-2).numpy()), axis=0)
                n_prop = np.concatenate((n_prop, prd_prop.detach().cpu().numpy()), axis=0)
        cell_type_idx_dic = context.cell_type_idx_dic
        prd_list = list(
            map(lambda x: cell_type_idx_dic.getGene(x) if cell_type_idx_dic.getGene(x) else 'Unknow', y_prd_list))

        context.prop = n_prop
        context.n_embedding = n_embedding
        context.cell_type_prd_list = prd_list

    def show_attention_weights(self, context: Optional[Context]):

        self.model.eval()
        n_embedding = None
        n_attention_weights, n_label, n_feature_idx = None, None, None
        n_true_label = None
        target_loader = context.target_loader
        for i, batch in enumerate(target_loader):

            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad, feature_idx_lens, \
                feature_val_lens, label = batch
            key_padding_mask = torch.zeros_like(feature_idx_pad).to(self.device)
            key_padding_mask[feature_idx_pad == 0] = 1
            key_padding_mask = key_padding_mask.unsqueeze(-2)
            tissue_idx = tissue_idx.unsqueeze(-1)
            tissue_val = tissue_val.unsqueeze(-1).unsqueeze(-1)
            feature_val_pad = feature_val_pad.unsqueeze(-1)
            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad = tissue_idx.to(self.device), \
                tissue_val.to(self.device), feature_idx_pad.to(self.device), feature_val_pad.to(self.device)
            label = label.to(self.device).unsqueeze(-1)
            batch_size = label.shape[0]

            embedding, attention_weights = self.model.encode(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad,
                                                             feature_val_pad,
                                                             key_padding_mask)
            prd = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad, feature_val_pad,
                             key_padding_mask, None)
            prd_prop = torch.softmax(prd, dim=-1)
            tmp_prd = torch.flatten(torch.argmax(prd_prop, dim=-1)).cpu().numpy()

            # attention_weights = attention_weights.squeeze(-2)
            feature_idx_pad = feature_idx_pad.cpu()
            print(f'feature idx shape: {feature_idx_pad.shape}')
            attention_weights = attention_weights.squeeze(-2).detach().cpu().numpy()
            print(attention_weights.sum(-1))
            print(f'attention weights shape: {attention_weights.shape}')
            # embedding, _ = self.model.encode(tissue_idx, tissue_val, feature_idx_pad, feature_val_pad,
            #                                  key_padding_mask)
            current_gene_num = attention_weights.shape[-1]
            cell_attention_tmp = np.zeros((batch_size, context.gene_num), dtype=np.float32)
            print(f'cell attention tmp shape: {cell_attention_tmp.shape}')
            cell_idx = np.arange(batch_size).reshape(1, -1).repeat(current_gene_num, axis=0)
            cell_idx = np.transpose(cell_idx)
            print(f'cell_idx shape: {cell_idx.shape}')
            print(cell_attention_tmp[cell_idx, feature_idx_pad].shape)

            cell_attention_tmp[cell_idx, feature_idx_pad] = attention_weights
            print(attention_weights.sum(-1))
            print(cell_attention_tmp.sum(-1))
            print(f'cell attention tmp shape: {cell_attention_tmp.shape}')

            if n_embedding is None:
                n_embedding = embedding.detach().cpu().squeeze(-2).numpy()
                n_label = tmp_prd
                n_true_label = label.cpu().numpy()
                n_attention_weights = cell_attention_tmp
                # n_label = tissue_idx.cpu().numpy()
            else:
                n_embedding = np.concatenate((n_embedding, embedding.detach().cpu().squeeze(-2).numpy()), axis=0)
                n_label = np.concatenate((n_label, tmp_prd), axis=0)
                n_true_label = np.concatenate((n_true_label, label.cpu().numpy()), axis=0)
                n_attention_weights = np.concatenate((n_attention_weights, cell_attention_tmp), axis=0)
                # n_label = np.concatenate((n_label, tissue_idx.cpu().numpy()), axis=0)
        print(f'n attention weights: {n_attention_weights.shape}')
        print(n_attention_weights.sum(-1))

        n_label = n_label.flatten()

        attention_weights_adata = sc.AnnData(X=n_attention_weights)
        # attention_weights_adata.obs['cell_type_prd'] = n_label
        # attention_weights_adata.obs['cell_type_true'] = n_true_label
        context.attention_weights_adata = attention_weights_adata
        # write_to_h5ad(attention_weights_adata, f'interpretable/{context.dataset_name}_attnetion.h5ad')