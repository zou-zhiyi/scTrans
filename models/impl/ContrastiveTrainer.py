import gc
import scanpy as sc
import glob
import math
import os
import time
from collections import Counter

import pandas as pd
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from models.dataset import generate_dataset_list, PadCollate, generate_dataset_from_pk, \
    generate_dataset_list_with_hvg, generate_dataset_list_no_celltype, PadCollate_no_celltype
from models.model import ContrastiveLoss, embedding_pca_initializing, \
    LabelSmoothing, generate_enhance_core_model, \
    embedding_pca_initializing_from_pk
from models.train import Trainer, Context
from models.utils import set_seed, check_anndata, read_mapping_file


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

    def __init__(self, model: nn.Module, train_dataset, test_dataset,save_path='', device_name='cpu', lr=0.001,
                 weight_decay=1e-2, trained_model_path=None, continue_train=False):
        super().__init__(model, train_dataset, test_dataset, lr=lr, device_name=device_name,
                         weight_decay=weight_decay, trained_model_path=trained_model_path,
                         continue_train=continue_train, save_path=save_path)
        self.loss = ContrastiveLoss(t=0.1)

    def train_inner(self, train_loader, context: Optional[Context]):
        loss_sum = []
        for i, batch in enumerate(train_loader):
            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad, feature_idx_lens, \
                feature_val_lens = batch

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
            # print(feature_idx_pad_1.shape)
            prd_1, _, _ = self.model.encode_with_projection_head(torch.ones_like(tissue_idx), tissue_val,
                                                                 feature_idx_pad_1,
                                                                 feature_val_pad_1,
                                                                 key_padding_mask, None)

            feature_idx_pad_2 = torch.clone(feature_idx_pad)
            feature_val_pad_2 = torch.clone(feature_val_pad)

            key_padding_mask = torch.zeros_like(feature_idx_pad_2).to(self.device)
            key_padding_mask[feature_idx_pad_2 == 0] = 1
            key_padding_mask = key_padding_mask.unsqueeze(-2)
            feature_val_pad_2 = feature_val_pad_2.unsqueeze(-1)
            prd_2, _, _ = self.model.encode_with_projection_head(torch.ones_like(tissue_idx), tissue_val,
                                                                 feature_idx_pad_2,
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

    def generate_new_embedding(self, context: Optional[Context]):
        batch_size = context.batch_size
        pad_collate = context.pad_collate
        data_loader = context.data_loader
        self.model.eval()
        n_embedding, n_label = None, None
        y_prd_list = []
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

            embedding, attn_score = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad,
                                               feature_val_pad,
                                               key_padding_mask)

            if n_embedding is None:
                n_embedding = embedding.detach().cpu().squeeze(-2).numpy()
                # n_label = tissue_idx.cpu().numpy()
            else:
                n_embedding = np.concatenate((n_embedding, embedding.detach().cpu().squeeze(-2).numpy()), axis=0)
        context.n_embedding = n_embedding

    def show_attention_weights(self, context: Optional[Context]):
        # batch_size = context.batch_size
        # pad_collate = context.pad_collate
        # visual_save_path = context.visual_save_path
        # data_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=False,
        #                          collate_fn=pad_collate)
        # cell_num = len(self.train_dataset)
        # cell_attention = np.zeros_like((cell_num, context.gene_num), dtype=float)
        self.model.eval()
        n_embedding = None
        n_attention_weights, n_feature_idx = None, None
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
            batch_size = tissue_idx.shape[0]

            embedding, attention_weights = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad,
                                                             feature_val_pad,
                                                             key_padding_mask)

            # attention_weights = attention_weights.squeeze(-2)
            feature_idx_pad = feature_idx_pad.cpu()
            # print(f'feature idx shape: {feature_idx_pad.shape}')
            attention_weights = attention_weights.squeeze(-2).detach().cpu().numpy()
            # print(attention_weights.sum(-1))
            # print(f'attention weights shape: {attention_weights.shape}')
            # embedding, _ = self.model.encode(tissue_idx, tissue_val, feature_idx_pad, feature_val_pad,
            #                                  key_padding_mask)
            current_gene_num = attention_weights.shape[-1]
            cell_attention_tmp = np.zeros((batch_size, context.gene_num), dtype=np.float32)
            # print(f'cell attention tmp shape: {cell_attention_tmp.shape}')
            cell_idx = np.arange(batch_size).reshape(1, -1).repeat(current_gene_num, axis=0)
            cell_idx = np.transpose(cell_idx)

            cell_attention_tmp[cell_idx, feature_idx_pad] = attention_weights


            if n_embedding is None:
                n_embedding = embedding.detach().cpu().squeeze(-2).numpy()
                n_attention_weights = cell_attention_tmp
            else:
                n_embedding = np.concatenate((n_embedding, embedding.detach().cpu().squeeze(-2).numpy()), axis=0)
                n_attention_weights = np.concatenate((n_attention_weights, cell_attention_tmp), axis=0)
        context.n_attention_weights= n_attention_weights


def show_embedding(dataset_filepath, title_name, d_model, head, d_ffw, dropout_rate, mapping_file, enhance_num,
                   vocab, batch_size, lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,
                   save_model_path=None, trained_model_path=None, continue_train=False, visual_save_path='empty',
                   anndata_postfix=''):
    set_seed(random_seed)
    word_idx_dic, _ = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset_list, adata_list \
        = generate_dataset_list_no_celltype(filepath_list=dataset_filepath, word_idx_dic=word_idx_dic)
    embedding_data_name = []
    for filepath in dataset_filepath:
        filename, _ = os.path.splitext(os.path.basename(filepath))
        embedding_data_name.append(filename)
    batch_set = 0
    # for adata in adata_list:
    # batch_set = set(adata.obs['batch_id'])
    batch_set = [0]
    print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    print(f'batch set num: {len(batch_set)}')
    print(f'batch set: {batch_set}')
    model = generate_enhance_core_model(d_model=d_model, h_dim=d_model, head=head, d_ffw=d_ffw,
                                        dropout_rate=dropout_rate,
                                        vocab=vocab, device_name=device_name, mlp_layer=project_head_layer,
                                        enhance_num=enhance_num)
    if trained_model_path is not None:
        state_dict = torch.load(trained_model_path)
        model.load_state_dict(state_dict['model'])

    # train_dataset = ConcatDataset(train_dataset_list)

    total_embedding_list = []
    for i in range(len(train_dataset_list)):
        trainer = ContrastiveTrainer(model, train_dataset_list[i], [], continue_train=continue_train,
                                     trained_model_path=None, device_name=device_name, lr=lr)

        ctx = Context(batch_size=batch_size, save_model_path=save_model_path,
                      pad_collate=PadCollate_no_celltype(), random_seed=None, epoch=None)
        ctx.visual_save_path = visual_save_path + "_" + embedding_data_name[i]
        ctx.title_name = title_name
        ctx.embedding_data_name = embedding_data_name[i]
        trainer.generate_new_embedding(ctx)
        total_embedding_list.append(ctx.n_embedding)
        # total_true_list.append(ctx.true_list)
        # total_prd_list.append(ctx.prd_list)
        # total_prop_list.append(ctx.prop)
        new_adata = sc.AnnData(X=ctx.n_embedding)
        new_adata.write_h5ad(f'interpretable/embedding/{ctx.embedding_data_name}{anndata_postfix}.h5ad')


def train_enhance_contrastive_model(train_filepath, epoch, d_model, h_dim, head, d_ffw, dropout_rate,
                                    mapping_file, vocab, pca_num, batch_size, enhance_num, with_d=True,
                                    lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,embedding_dropout=False,
                                    save_model_path=None, trained_model_path=None, continue_train=False, freeze=False):
    set_seed(random_seed)
    word_idx_dic, _ = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset_list, adata_list \
        = generate_dataset_list_no_celltype(filepath_list=train_filepath, word_idx_dic=word_idx_dic)
    # print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    model = generate_enhance_core_model(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw,
                                        dropout_rate=dropout_rate,
                                        vocab=vocab, device_name=device_name, mlp_layer=project_head_layer,
                                        enhance_num=enhance_num, embedding_dropout=embedding_dropout)

    if trained_model_path is None:
        embedding_pca_initializing(model, adata_list, pca_num, word_idx_dic, device_name, vocab)
    if freeze:
        model.embedding.requires_grad_(False)
    gc.collect()

    total_train_dataset = ConcatDataset(train_dataset_list)
    print(f"train dataset size:{len(total_train_dataset)}")

    trainer = ContrastiveTrainer(model, [total_train_dataset], [], continue_train=continue_train,
                                 trained_model_path=trained_model_path, device_name=device_name, lr=lr)

    ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate_no_celltype(), random_seed=None)
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
                                    d_model=64, h_dim=64, head=head_num, d_ffw=64 * 3, dropout_rate=0.2, vocab=40000,
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
