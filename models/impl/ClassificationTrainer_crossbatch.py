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

from datasets.preprocess import read_mapping_file
from models.dataset import generate_train_test_dataset_list, PadCollate, \
    generate_dataset_list, generate_dataset_list_with_hvg
from models.impl.ContrastiveTrainer import train_enhance_contrastive, train_enhance_contrastive_model
from models.model import LabelSmoothing, generate_enhance_classification_model_with_d
from models.train import Trainer, Context, enhance_classification_construct

from models.utils import set_seed, calculate_score, tsne_plot, umap_plot, write_file_to_pickle, scatter_plot, \
    leiden_clustering, draw_pie_scatter, check_anndata, write_to_h5ad

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
                 weight_decay=1e-2, trained_model_path=None, continue_train=False):
        super().__init__(model, train_dataset, test_dataset, lr=lr, device_name=device_name,
                         weight_decay=weight_decay, trained_model_path=trained_model_path,
                         continue_train=continue_train)
        self.threshold = 0.5

    def train_inner(self, train_loader, context: Optional[Context]):
        cross_entropy_loss = LabelSmoothing(0)
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

            softmax_loss = cross_entropy_loss(prd, label)

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
        cross_entropy_loss = LabelSmoothing(0)
        loss_sum = []
        cnt, total = 0, 0
        y_prop_list = None
        y_prd_list = []
        y_true_list = []
        auc = 0
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
            softmax_loss = cross_entropy_loss(prd, label)

            loss_sum.append(softmax_loss.item())


        acc, ari, f1_scores_median, f1_scores_macro, f1_scores_micro, f1_scores_weighted = calculate_score(y_true_list,
                                                                                                           y_prd_list)
        context.acc_list.append(acc)
        context.auc_list.append(auc)
        context.ari_list.append(ari)
        context.f1_scores_median_list.append(f1_scores_median)
        context.f1_scores_macro_list.append(f1_scores_macro)
        context.f1_scores_micro_list.append(f1_scores_micro)
        context.f1_scores_weighted_list.append(f1_scores_weighted)

        context.best_acc = max(context.best_acc, acc)
        context.last_acc = acc
        context.best_auc = max(context.best_auc, auc)
        context.last_auc = auc
        context.best_ari = max(context.best_ari, ari)
        context.last_ari = ari
        context.best_f1_scores_median = max(context.best_f1_scores_median, f1_scores_median)
        context.last_f1_scores_median = f1_scores_median
        context.best_f1_scores_macro = max(context.best_f1_scores_macro, f1_scores_macro)
        context.last_f1_scores_macro = f1_scores_macro
        context.best_f1_scores_micro = max(context.best_f1_scores_micro, f1_scores_micro)
        context.last_f1_scores_micro = f1_scores_micro
        context.best_f1_scores_weighted = max(context.best_f1_scores_weighted, f1_scores_weighted)
        context.last_f1_scores_weighted = f1_scores_weighted

        context.epoch_loss = np.mean(loss_sum)
        context.cell_type_prd_list = y_prd_list
        context.cell_type_true_list = y_true_list

    def generate_new_embedding(self, context: Optional[Context]):
        batch_size = context.batch_size
        pad_collate = context.pad_collate
        color_map = context.color_map
        data_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=pad_collate)
        self.model.eval()
        n_embedding, n_label, n_prop = None, None, None
        y_prd_list = []
        for i, batch in enumerate(data_loader):
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

            embedding, attn_score = self.model.encode(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad,
                                                      feature_val_pad,
                                                      key_padding_mask)

            prd = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad, feature_val_pad,
                             key_padding_mask, None)

            prd_prop = torch.softmax(prd, dim=-1)

            pd_max_prop, _ = torch.max(prd_prop, dim=-1)
            tmp_prd = torch.flatten(torch.argmax(prd_prop, dim=-1)).cpu().numpy()

            y_prd_list = y_prd_list + tmp_prd.tolist()

            if n_embedding is None:
                n_embedding = embedding.detach().cpu().squeeze(-2).numpy()
                n_label = label.cpu().numpy()
                n_prop = prd_prop.detach().cpu().numpy()
                # n_label = tissue_idx.cpu().numpy()
            else:
                n_embedding = np.concatenate((n_embedding, embedding.detach().cpu().squeeze(-2).numpy()), axis=0)
                n_label = np.concatenate((n_label, label.cpu().numpy()), axis=0)
                n_prop = np.concatenate((n_prop, prd_prop.detach().cpu().numpy()), axis=0)
                # n_label = np.concatenate((n_label, tissue_idx.cpu().numpy()), axis=0)


        n_label = n_label.flatten()

        # print(n_embedding.shape)
        # print(n_label.shape)

        cell_type_idx_dic = context.word_idx_dic
        prd_list = list(
            map(lambda x: cell_type_idx_dic.getGene(x) if cell_type_idx_dic.getGene(x) else 'Unknow', y_prd_list))
        true_list = list(
            map(lambda x: cell_type_idx_dic.getGene(x) if cell_type_idx_dic.getGene(x) else 'Unknow', n_label.tolist()))

        context.color_map = color_map
        context.prop = n_prop
        context.n_embedding = n_embedding
        context.true_list = true_list
        context.prd_list = prd_list

    def show_attention_weights(self, context: Optional[Context]):
        batch_size = context.batch_size
        pad_collate = context.pad_collate
        # visual_save_path = context.visual_save_path
        data_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=pad_collate)
        cell_num = len(self.train_dataset)
        # cell_attention = np.zeros_like((cell_num, context.gene_num), dtype=float)
        self.model.eval()
        n_embedding = None
        n_attention_weights, n_label, n_feature_idx = None, None, None
        n_true_label = None
        for i, batch in enumerate(data_loader):

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
        attention_weights_adata.obs['cell_type_prd'] = n_label
        attention_weights_adata.obs['cell_type_true'] = n_true_label
        write_to_h5ad(attention_weights_adata, f'interpretable/{context.dataset_name}_attnetion.h5ad')

def show_embedding(dataset_filepath, title_name, d_model, head, d_ffw, dropout_rate, mapping_file, enhance_num,
                   mlp_layer,
                   vocab, batch_size, lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,
                   save_model_path=None, trained_model_path=None, continue_train=False, visual_save_path='empty',
                   anndata_postfix=''):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    predict_type = len(cell_type_idx_dic.word2idx_dic)
    train_dataset_list, adata_list \
        = generate_dataset_list(filepath_list=dataset_filepath, word_idx_dic=word_idx_dic,
                                cell_type_idx_dic=cell_type_idx_dic)
    embedding_data_name = []
    for filepath in dataset_filepath:
        filename, _ = os.path.splitext(os.path.basename(filepath))
        embedding_data_name.append(filename)
    model = generate_enhance_classification_model_with_d(d_model=d_model, h_dim=d_model, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=predict_type, vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num)
    if trained_model_path is not None:
        state_dict = torch.load(trained_model_path)
        model.encode.load_state_dict(state_dict['model'])
    color_map = None

    train_dataset = ConcatDataset(train_dataset_list)
    trainer = ClassificationTrainer(model, train_dataset, [], continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)

    ctx = Context(batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None, epoch=None)
    ctx.word_idx_dic = cell_type_idx_dic
    ctx.color_map = color_map
    ctx.embedding_data_name = 'Total'
    trainer.generate_new_embedding(ctx)
    color_map = ctx.color_map

    total_embedding_list = []
    total_true_list = []
    total_prd_list = []
    total_prop_list = []
    for i in range(len(train_dataset_list)):
        trainer = ClassificationTrainer(model, train_dataset_list[i], [], continue_train=continue_train,
                                        trained_model_path=None, device_name=device_name, lr=lr)

        ctx = Context(batch_size=batch_size, save_model_path=save_model_path,
                      pad_collate=PadCollate(), random_seed=None, epoch=None)
        ctx.word_idx_dic = cell_type_idx_dic
        ctx.color_map = color_map
        ctx.visual_save_path = visual_save_path + "_" + embedding_data_name[i]
        ctx.title_name = title_name
        ctx.embedding_data_name = embedding_data_name[i]
        trainer.generate_new_embedding(ctx)
        color_map = ctx.color_map
        total_embedding_list.append(ctx.n_embedding)
        total_true_list.append(ctx.true_list)
        total_prd_list.append(ctx.prd_list)
        total_prop_list.append(ctx.prop)
        # if ctx.embedding_data_name == title_name:
        new_adata = sc.AnnData(X=ctx.n_embedding)
        new_adata.obs['cell_type_prd'] = ctx.prd_list
        new_adata.obs['cell_type_true'] = ctx.true_list
        new_adata.write_h5ad(f'interpretable/embedding/{ctx.embedding_data_name}{anndata_postfix}.h5ad')


def show_attention_weights(dataset_filepath, d_model, h_dim, head, d_ffw, dropout_rate, mapping_file, enhance_num,
                           mlp_layer,
                           vocab, batch_size, lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,
                           save_model_path=None, trained_model_path=None, continue_train=False, dataset_name='empty',
                           k=1000):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    predict_type = len(cell_type_idx_dic.word2idx_dic)
    train_dataset_list, adata_list \
        = generate_dataset_list(filepath_list=dataset_filepath, word_idx_dic=word_idx_dic,
                                cell_type_idx_dic=cell_type_idx_dic)
    model = generate_enhance_classification_model_with_d(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=predict_type, vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num)
    if trained_model_path is not None:
        state_dict = torch.load(trained_model_path)
        model.load_state_dict(state_dict['model'])
    trainer = ClassificationTrainer(model, train_dataset_list[0], [], continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)

    ctx = Context(batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None, epoch=None)
    ctx.gene_num = len(word_idx_dic.word2idx_dic)
    ctx.word_idx_dic = word_idx_dic
    ctx.dataset_name = dataset_name
    ctx.k = k
    trainer.show_attention_weights(ctx)


def train_enhance_class_model_with_extra(train_filepath, test_filepath, batch_size,
                                         mapping_file, epoch, d_model, h_dim, head, d_ffw, dropout_rate,
                                         vocab, continue_train, save_model_path, pca_num, freeze,
                                         lr=0.001, device_name='cpu', random_seed=None, mlp_layer=None,
                                         enhance_num=1, trained_model_path=None, embedding_dropout=False):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    print(cell_type_idx_dic.word2idx_dic)
    predict_type = len(cell_type_idx_dic.word2idx_dic)
    train_dataset_list, train_adata_list = \
        generate_dataset_list(filepath_list=train_filepath, word_idx_dic=word_idx_dic,
                              cell_type_idx_dic=cell_type_idx_dic)
    test_dataset_list, test_adata_list = \
        generate_dataset_list(filepath_list=test_filepath, word_idx_dic=word_idx_dic,
                              cell_type_idx_dic=cell_type_idx_dic)
    print(cell_type_idx_dic.word2idx_dic)
    model = generate_enhance_classification_model_with_d(d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=predict_type, vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num, embedding_dropout=embedding_dropout)

    if trained_model_path is None:
        enhance_classification_construct(model, train_adata_list, pca_num, word_idx_dic, device_name, vocab)
    else:
        model_state = torch.load(trained_model_path)
        model.encode.load_state_dict(model_state['model'])

    if freeze:
        model.encode.embedding.requires_grad_(False)

    total_train_dataset = ConcatDataset(train_dataset_list)
    total_test_dataset = ConcatDataset(test_dataset_list)
    gc.collect()

    trainer = ClassificationTrainer(model, total_train_dataset, test_dataset=None, continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)
    ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None)

    ctx.predict_type = predict_type

    ctx.acc_list, ctx.ari_list, ctx.f1_scores_median_list, ctx.f1_scores_macro_list, ctx.f1_scores_micro_list, \
        ctx.f1_scores_weighted_list = [], [], [], [], [], []
    ctx.auc_list, ctx.best_auc, ctx.last_auc = [], 0, 0
    ctx.best_acc, ctx.best_ari, ctx.best_f1_scores_median, ctx.best_f1_scores_macro, ctx.best_f1_scores_micro, \
        ctx.best_f1_scores_weighted = 0, 0, 0, 0, 0, 0
    ctx.last_acc, ctx.last_ari, ctx.last_f1_scores_median, ctx.last_f1_scores_macro, ctx.last_f1_scores_micro, \
        ctx.last_f1_scores_weighted = 0, 0, 0, 0, 0, 0
    trainer.train(ctx)

    trainer.model.eval()
    with torch.no_grad():
        ctx =  Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None)
        ctx.predict_type = predict_type
        ctx.acc_list, ctx.ari_list, ctx.f1_scores_median_list, ctx.f1_scores_macro_list, ctx.f1_scores_micro_list, \
            ctx.f1_scores_weighted_list = [], [], [], [], [], []
        ctx.auc_list, ctx.best_auc, ctx.last_auc = [], 0, 0
        ctx.best_acc, ctx.best_ari, ctx.best_f1_scores_median, ctx.best_f1_scores_macro, ctx.best_f1_scores_micro, \
            ctx.best_f1_scores_weighted = 0, 0, 0, 0, 0, 0
        ctx.last_acc, ctx.last_ari, ctx.last_f1_scores_median, ctx.last_f1_scores_macro, ctx.last_f1_scores_micro, \
            ctx.last_f1_scores_weighted = 0, 0, 0, 0, 0, 0
        test_loader = DataLoader(dataset=total_test_dataset, batch_size=batch_size, shuffle=True,
                                 pin_memory=False, collate_fn=ctx.pad_collate, drop_last=False)
        trainer.test_inner(test_loader, ctx)

    return ctx


