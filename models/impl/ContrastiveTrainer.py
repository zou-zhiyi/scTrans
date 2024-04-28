import gc

import glob
import math
import time
from collections import Counter

import pandas as pd
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from datasets.preprocess import read_mapping_file
from models.dataset import generate_dataset_list, PadCollate, generate_dataset_from_pk, \
    generate_dataset_list_with_hvg
from models.model import ContrastiveLoss , embedding_pca_initializing, \
     LabelSmoothing, generate_enhance_core_model, \
    embedding_pca_initializing_from_pk
from models.train import Trainer, Context
from models.utils import set_seed, tsne_plot, umap_plot, check_anndata


def feature_mask_enhance(feature_idx_pad, mask_rate):
    mask = torch.zeros_like(feature_idx_pad, dtype=torch.float32).to(feature_idx_pad.device)
    mask[feature_idx_pad != 0] = mask_rate
    mask = torch.bernoulli(mask).long()
    return feature_idx_pad * mask


# not use
def feature_val_shuffle_enhance(feature_idx_pad, shuffle_rate):
    new_feature_idx_pad = torch.clone(feature_idx_pad)
    num = new_feature_idx_pad.shape[1]
    idx = torch.randperm(num)
    idx_mask = idx < (math.ceil(num * shuffle_rate))
    idx_mask2 = idx >= (num - math.ceil(num * shuffle_rate))
    tmp_feature = new_feature_idx_pad[:, idx_mask]
    new_feature_idx_pad[:, idx_mask] = new_feature_idx_pad[:, idx_mask2]
    new_feature_idx_pad[:, idx_mask2] = tmp_feature
    return new_feature_idx_pad


# not use
def feature_val_gs_noise_enhance(feature_val_pad, mean=0, std=0.3):
    new_feature_val_pad = torch.clone(feature_val_pad)
    noise = torch.normal(mean=mean, std=std, size=new_feature_val_pad.shape).to(new_feature_val_pad.device)
    new_feature_val_pad = new_feature_val_pad + noise
    new_feature_val_pad = torch.clip(new_feature_val_pad, 0)
    return new_feature_val_pad


# not use
def feature_val_mean_enhance(feature_val_pad):
    new_feature_val_pad = torch.clone(feature_val_pad)
    new_feature_val_pad[new_feature_val_pad != 0] = 1
    return new_feature_val_pad


class ContrastiveTrainer(Trainer):

    def __init__(self, model: nn.Module, train_dataset, test_dataset, device_name='cpu', lr=0.001,
                 weight_decay=1e-2, trained_model_path=None, continue_train=False):
        super().__init__(model, train_dataset, test_dataset, lr=lr, device_name=device_name,
                         weight_decay=weight_decay, trained_model_path=trained_model_path,
                         continue_train=continue_train)
        self.loss = ContrastiveLoss(t=0.1)

    def train_inner(self, train_loader, context: Optional[Context]):
        loss_sum = []
        for i, batch in enumerate(train_loader):

            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad, feature_idx_lens, \
                feature_val_lens, label = batch

            tissue_idx = tissue_idx.unsqueeze(-1)
            tissue_val = tissue_val.unsqueeze(-1).unsqueeze(-1)

            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad = tissue_idx.to(self.device), \
                tissue_val.to(self.device), feature_idx_pad.to(self.device), feature_val_pad.to(self.device)

            feature_idx_pad_1 = torch.clone(feature_idx_pad)
            feature_val_pad_1 = torch.clone(feature_val_pad)
            feature_idx_pad_1 = feature_mask_enhance(feature_idx_pad_1, 0.85)
            key_padding_mask = torch.zeros_like(feature_idx_pad_1).to(self.device)
            key_padding_mask[feature_idx_pad_1 == 0] = 1
            key_padding_mask = key_padding_mask.unsqueeze(-2)

            feature_val_pad_1 = feature_val_pad_1.unsqueeze(-1)
            prd_1, _, _ = self.model.encode_with_projection_head(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad_1,
                                                             feature_val_pad_1,
                                                             key_padding_mask, None)

            feature_idx_pad_2 = torch.clone(feature_idx_pad)
            feature_val_pad_2 = torch.clone(feature_val_pad)

            key_padding_mask = torch.zeros_like(feature_idx_pad_2).to(self.device)
            key_padding_mask[feature_idx_pad_2 == 0] = 1
            key_padding_mask = key_padding_mask.unsqueeze(-2)
            feature_val_pad_2 = feature_val_pad_2.unsqueeze(-1)
            prd_2, _, _ = self.model.encode_with_projection_head(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad_2,
                                                      feature_val_pad_2,
                                                      key_padding_mask, None)

            total_loss = self.loss(prd_1, prd_2)
            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()
            loss_sum.append(total_loss.item())
        context.epoch_loss = np.mean(loss_sum)

    def test_inner(self, test_loader, context: Optional[Context]):
        context.epoch_loss = 0

def train_enhance_contrastive_model(train_filepath, epoch, d_model, h_dim, head, d_ffw, dropout_rate,
                                    mapping_file, vocab, pca_num, batch_size, enhance_num, with_d=True,
                                    lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,
                                    save_model_path=None, trained_model_path=None, continue_train=False, freeze=False):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset_list, adata_list \
        = generate_dataset_list(filepath_list=train_filepath, word_idx_dic=word_idx_dic,
                                cell_type_idx_dic=cell_type_idx_dic)
    print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    model = generate_enhance_core_model(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw,
                                        dropout_rate=dropout_rate,
                                        vocab=vocab, device_name=device_name, mlp_layer=project_head_layer,
                                        enhance_num=enhance_num)

    if trained_model_path is None:
        embedding_pca_initializing(model, adata_list, pca_num, word_idx_dic, device_name, vocab)
    if freeze:
        model.embedding.requires_grad_(False)
    gc.collect()

    total_train_dataset = ConcatDataset(train_dataset_list)

    trainer = ContrastiveTrainer(model, [total_train_dataset], [], continue_train=continue_train,
                                 trained_model_path=trained_model_path, device_name=device_name, lr=lr)

    ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None)
    ctx.with_d = with_d
    trainer.train(ctx)

def train_enhance_contrastive(dir_name='mouse', dataset_name='Bone_marrow', word_dic_prefix='Bone_marrow',
                              cell_type_prefix='Bone_marrow', enhance_num=1, head_num=1, random_seed=None):
    data_files = glob.glob(f'../../datasets/{dir_name}/{dataset_name}/*.h5ad')
    print(list(data_files))
    f = list(data_files)
    train_enhance_contrastive_model(train_filepath=f,
                                    with_d=False,
                                    freeze=False,
                                    epoch=40,
                                    lr=0.0002,
                                    enhance_num=enhance_num,
                                    project_head_layer=None,
                                    mapping_file=[
                                        f'../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                                        f'../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
                                    save_model_path=f'pretrained/{dataset_name}_enhance{enhance_num}_{head_num}head_pretrained_cts_model.pth',
                                    d_model=64, h_dim=64, head=head_num, d_ffw=64*3, dropout_rate=0.2, vocab=40000,
                                    pca_num=64,
                                    batch_size=100,
                                    device_name='cuda:0', random_seed=random_seed, continue_train=False)
    print(list(data_files))


def train_enhance_contrastive_model_from_pk(train_filepath, pca_adata_filepath, epoch, d_model, head, d_ffw,
                                            dropout_rate,
                                            mapping_file, vocab, pca_num, batch_size, enhance_num,
                                            lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,
                                            save_model_path=None, trained_model_path=None, continue_train=False,
                                            freeze=False):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset = generate_dataset_from_pk(train_filepath)
    batch_set = [1]
    batch_set = set(batch_set)
    print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    model = generate_enhance_core_model(d_model=d_model, h_dim=d_model, head=head, d_ffw=d_ffw,
                                        dropout_rate=dropout_rate,
                                        vocab=vocab, device_name=device_name, mlp_layer=project_head_layer,
                                        enhance_num=enhance_num)

    adata = check_anndata(pca_adata_filepath)
    if trained_model_path is None:
        embedding_pca_initializing_from_pk(model.enhance_model_core, [adata], pca_num, word_idx_dic, device_name, vocab)
    if freeze:
        model.enhance_model_core.embedding.requires_grad_(False)
    gc.collect()
    # total_train_dataset = ConcatDataset(train_dataset_list)
    # total_test_dataset = ConcatDataset(test_dataset_list)

    trainer = ContrastiveTrainer(model, [train_dataset], [], continue_train=continue_train,
                                 trained_model_path=trained_model_path, device_name=device_name, lr=lr)

    ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None)
    trainer.train(ctx)
    del ctx
    del trainer
    del model
    del train_dataset
    del adata
    gc.collect()


def train_enhance_contrastive_from_pk(pk_filepath, adata_filepath, enhance_num, dir_name, word_dic_prefix,
                                      cell_type_prefix,
                                      dataset_name, head, random_seed=None, epoch=40):
    train_enhance_contrastive_model_from_pk(train_filepath=pk_filepath,
                                            pca_adata_filepath=adata_filepath,
                                            freeze=False,
                                            epoch=epoch,
                                            lr=0.0002,
                                            enhance_num=enhance_num,
                                            project_head_layer=None,
                                            mapping_file=[
                                                f'../../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                                                f'../../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
                                            save_model_path=f'pretrained/{dataset_name}_tissue_enhance{enhance_num}_{head}head_pretrained_cts_model_300_percent_pe_mean_with_tissue_without_D.pth',
                                            d_model=64, head=head, d_ffw=64 * 3, dropout_rate=0.2, vocab=40000,
                                            pca_num=64,
                                            batch_size=100,
                                            device_name='cuda:0', random_seed=random_seed, continue_train=False)
