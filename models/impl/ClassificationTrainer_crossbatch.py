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
                             key_padding_mask)

            softmax_loss = cross_entropy_loss(prd, label)

            prd_prop = torch.softmax(prd, dim=-1)
            tmp_prd = torch.flatten(torch.argmax(prd_prop, dim=-1)).cpu().numpy()

            y_prd_list = y_prd_list + tmp_prd.tolist()
            y_true_list = y_true_list + torch.flatten(label).cpu().numpy().tolist()

            self.opt.zero_grad()
            softmax_loss.backward()
            self.opt.step()
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
            # prd = prd.unsqueeze(-2)
            # print(f'prd shape: {prd.shape}')
            # print(f'label shape: {label.shape}')
            # print()
            # mask_prd, mask_prd_indices = torch.max(prd, dim=-1)
            # mask_prd_indices = mask_prd_indices.squeeze(-1)
            # torch.index_select()
            # print(f'argmax shape:{torch.argmax(prd, dim=-1).shape}')
            # print((mask_prd_indices == mask_label).shape)
            # print(mask_prd > 0.95)
            # cnt_list = ((mask_prd_indices == label) & (mask_prd > self.threshold)).cpu().numpy().sum()

            # print(f'result shape: {(torch.argmax(prd, dim=-1) == label)}')
            # print(cnt_list)
            # cnt += cnt_list
            # total += len(label)

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

    def generate_new_embedding(self, context: Optional[Context]):
        batch_size = context.batch_size
        pad_collate = context.pad_collate
        title_name = context.title_name
        visual_save_path = context.visual_save_path
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
    batch_set = 0
    # for adata in adata_list:
    # batch_set = set(adata.obs['batch_id'])
    batch_set = [0]
    print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    # predict_type = adata.uns['cell_type_nums']
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    print(f'batch set num: {len(batch_set)}')
    print(f'batch set: {batch_set}')
    model = generate_enhance_classification_model_with_d(d_model=d_model, h_dim=d_model, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=predict_type, vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num)
    if trained_model_path is not None:
        state_dict = torch.load(trained_model_path)
        model.load_state_dict(state_dict['model'])
    color_map = None

    train_dataset = ConcatDataset(train_dataset_list)
    trainer = ClassificationTrainer(model, train_dataset, [], continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)

    ctx = Context(batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None, epoch=None)
    ctx.word_idx_dic = cell_type_idx_dic
    ctx.color_map = color_map
    ctx.visual_save_path = visual_save_path + "_total"
    ctx.title_name = title_name
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
        if ctx.embedding_data_name == title_name:
            new_adata = sc.AnnData(X=ctx.n_embedding)
            new_adata.obs['cell_type_prd'] = ctx.prd_list
            new_adata.obs['cell_type_true'] = ctx.true_list
            new_adata.write_h5ad(f'interpretable/embedding/{ctx.embedding_data_name}{anndata_postfix}.h5ad')
            # color_map = tsne_plot(data=ctx.n_embedding, label_name=ctx.prd_list, color_map=color_map,
            #                       save_file_name=f"interpretable/embedding/hvg2000/{visual_save_path}-{ctx.embedding_data_name}{anndata_postfix}-cell_type_prd.jpg")
            # color_map = tsne_plot(data=ctx.n_embedding, label_name=ctx.true_list, color_map=color_map,
            #                       save_file_name=f"interpretable/embedding/hvg2000/{visual_save_path}-{ctx.embedding_data_name}{anndata_postfix}-cell_type_true.jpg")
            # nmi_score = normalized_mutual_info_score(ctx.true_list, best_cluster_result)
            # ari_score = adjusted_rand_score(ctx.true_list, best_cluster_result)
            # _ = tsne_plot(data=ctx.n_embedding, label_name=np.array(best_cluster_result), color_map=None,
            #               save_file_name=f"interpretable/embedding/hvg2000/{visual_save_path}-{ctx.embedding_data_name}{anndata_postfix}-k-means.jpg",
            #               title_name=f'ARI:{round(ari_score, 2)}, NMI:{round(nmi_score, 2)}, SCI:{round(best_si + 0, 2)}')


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


def show_attention_weights_prd(dataset_filepath, d_model, h_dim, head, d_ffw, dropout_rate, mapping_file, enhance_num,
                               mlp_layer,
                               vocab, batch_size, lr=0.001, device_name='cpu', random_seed=None,
                               project_head_layer=None,
                               save_model_path=None, trained_model_path=None, continue_train=False,
                               dataset_name='empty',
                               k=1000):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    predict_type = len(cell_type_idx_dic.word2idx_dic)
    train_dataset_list, adata_list \
        = generate_dataset_list(filepath_list=dataset_filepath, word_idx_dic=word_idx_dic,
                                cell_type_idx_dic=cell_type_idx_dic)

    batch_set = [1]
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
    trainer.show_attention_weights_prd(ctx)

def train_enhance_class_model_with_d(train_filepath, test_size, epoch, d_model, head, d_ffw, dropout_rate,
                                     mapping_file,
                                     vocab, pca_num, batch_size, lr=0.001, device_name='cpu', random_seed=None,
                                     mlp_layer=None,
                                     save_model_path=None, trained_model_path=None, continue_train=False, freeze=False,
                                     project_layer=None, enhance_num=1, execute_model='train'):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset_list, test_dataset_list, adata_list \
        = generate_train_test_dataset_list(filepath_list=train_filepath, test_size=test_size,
                                           word_idx_dic=word_idx_dic, cell_type_idx_dic=cell_type_idx_dic,
                                           random_seed=random_seed)
    print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    model = generate_enhance_classification_model_with_d(d_model=d_model, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=len(cell_type_idx_dic.word2idx_dic), vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num)
    if trained_model_path is None:
        enhance_classification_construct(model, adata_list, pca_num, word_idx_dic, device_name, vocab)
    else:
        model_state = torch.load(trained_model_path)
        model.encode.load_state_dict(model_state['model'])
    if freeze:
        model.encode.embedding.requires_grad_(False)
    gc.collect()

    total_train_dataset = ConcatDataset(train_dataset_list)
    total_test_dataset = ConcatDataset(test_dataset_list)

    trainer = ClassificationTrainer(model, total_train_dataset, total_test_dataset, continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)

    ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None)
    ctx.execute_model = execute_model
    ctx.acc_list, ctx.ari_list, ctx.f1_scores_median_list, ctx.f1_scores_macro_list, ctx.f1_scores_micro_list, \
        ctx.f1_scores_weighted_list = [], [], [], [], [], []
    ctx.auc_list, ctx.best_auc, ctx.last_auc = [], 0, 0
    ctx.best_acc, ctx.best_ari, ctx.best_f1_scores_median, ctx.best_f1_scores_macro, ctx.best_f1_scores_micro, \
        ctx.best_f1_scores_weighted = 0, 0, 0, 0, 0, 0
    ctx.last_acc, ctx.last_ari, ctx.last_f1_scores_median, ctx.last_f1_scores_macro, ctx.last_f1_scores_micro, \
        ctx.last_f1_scores_weighted = 0, 0, 0, 0, 0, 0
    trainer.train(ctx)

    print(f'ctx acc list: {ctx.acc_list}')
    print(f'ctx auc list: {ctx.auc_list}')
    print(f'ctx ari list: {ctx.ari_list}')
    print(f'ctx f1_scores_median list: {ctx.f1_scores_median_list}')
    print(f'ctx f1_scores_macro list: {ctx.f1_scores_macro_list}')
    print(f'ctx f1_scores_micro list: {ctx.f1_scores_micro_list}')
    print(f'ctx f1_scores_weighted list: {ctx.f1_scores_weighted_list}')

    print(f'ctx best acc: {ctx.best_acc}')
    print(f'ctx best auc: {ctx.best_auc}')
    print(f'ctx best ari: {ctx.best_ari}')
    print(f'ctx best f1_scores_median: {ctx.best_f1_scores_median}')
    print(f'ctx best f1_scores_macro: {ctx.best_f1_scores_macro}')
    print(f'ctx best f1_scores_micro: {ctx.best_f1_scores_micro}')
    print(f'ctx best f1_scores_weighted: {ctx.best_f1_scores_weighted}')

    print(f'ctx last acc: {ctx.last_acc}')
    print(f'ctx last auc: {ctx.last_auc}')
    print(f'ctx last ari: {ctx.last_ari}')
    print(f'ctx last f1_scores_median: {ctx.last_f1_scores_median}')
    print(f'ctx last f1_scores_macro: {ctx.last_f1_scores_macro}')
    print(f'ctx last f1_scores_micro: {ctx.last_f1_scores_micro}')
    print(f'ctx last f1_scores_weighted: {ctx.last_f1_scores_weighted}')

    return ctx


def train_enhance_class_model_with_extra(train_filepath, test_filepath, batch_size,
                                         mapping_file, epoch, d_model, h_dim, head, d_ffw, dropout_rate,
                                         vocab, continue_train, save_model_path, pca_num, freeze,
                                         lr=0.001, device_name='cpu', random_seed=None, mlp_layer=None,
                                         enhance_num=1, trained_model_path=None):
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
                                                         enhance_num=enhance_num)

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

    trainer = ClassificationTrainer(model, total_train_dataset, total_test_dataset, continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)
    ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None)
    ctx.predict_type = predict_type

    total_acc_list, total_ari_list, total_f1_scores_median_list, total_f1_scores_macro_list, total_f1_scores_micro_list, \
        total_f1_scores_weighted_list = [], [], [], [], [], []
    total_auc_list = []

    ctx.acc_list, ctx.ari_list, ctx.f1_scores_median_list, ctx.f1_scores_macro_list, ctx.f1_scores_micro_list, \
        ctx.f1_scores_weighted_list = [], [], [], [], [], []
    ctx.auc_list, ctx.best_auc, ctx.last_auc = [], 0, 0
    ctx.best_acc, ctx.best_ari, ctx.best_f1_scores_median, ctx.best_f1_scores_macro, ctx.best_f1_scores_micro, \
        ctx.best_f1_scores_weighted = 0, 0, 0, 0, 0, 0
    ctx.last_acc, ctx.last_ari, ctx.last_f1_scores_median, ctx.last_f1_scores_macro, ctx.last_f1_scores_micro, \
        ctx.last_f1_scores_weighted = 0, 0, 0, 0, 0, 0
    trainer.train(ctx)

    print(f'ctx acc list: {ctx.acc_list}')
    print(f'ctx auc list: {ctx.auc_list}')
    print(f'ctx ari list: {ctx.ari_list}')
    print(f'ctx f1_scores_median list: {ctx.f1_scores_median_list}')
    print(f'ctx f1_scores_macro list: {ctx.f1_scores_macro_list}')
    print(f'ctx f1_scores_micro list: {ctx.f1_scores_micro_list}')
    print(f'ctx f1_scores_weighted list: {ctx.f1_scores_weighted_list}')

    print(f'ctx best acc: {ctx.best_acc}')
    print(f'ctx best auc: {ctx.best_auc}')
    print(f'ctx best ari: {ctx.best_ari}')
    print(f'ctx best f1_scores_median: {ctx.best_f1_scores_median}')
    print(f'ctx best f1_scores_macro: {ctx.best_f1_scores_macro}')
    print(f'ctx best f1_scores_micro: {ctx.best_f1_scores_micro}')
    print(f'ctx best f1_scores_weighted: {ctx.best_f1_scores_weighted}')

    print(f'ctx last acc: {ctx.last_acc}')
    print(f'ctx last auc: {ctx.last_auc}')
    print(f'ctx last ari: {ctx.last_ari}')
    print(f'ctx last f1_scores_median: {ctx.last_f1_scores_median}')
    print(f'ctx last f1_scores_macro: {ctx.last_f1_scores_macro}')
    print(f'ctx last f1_scores_micro: {ctx.last_f1_scores_micro}')
    print(f'ctx last f1_scores_weighted: {ctx.last_f1_scores_weighted}')

    total_acc_list.append(ctx.acc_list)
    total_ari_list.append(ctx.ari_list)
    total_auc_list.append(ctx.auc_list)
    total_f1_scores_median_list.append(ctx.f1_scores_median_list)
    total_f1_scores_macro_list.append(ctx.f1_scores_macro_list)
    total_f1_scores_micro_list.append(ctx.f1_scores_micro_list)
    total_f1_scores_weighted_list.append(ctx.f1_scores_weighted_list)

    for i in range(len(test_dataset_list)):
        ctx.acc_list, ctx.ari_list, ctx.f1_scores_median_list, ctx.f1_scores_macro_list, ctx.f1_scores_micro_list, \
            ctx.f1_scores_weighted_list = [], [], [], [], [], []
        ctx.auc_list, ctx.best_auc, ctx.last_auc = [], 0, 0
        ctx.best_acc, ctx.best_ari, ctx.best_f1_scores_median, ctx.best_f1_scores_macro, ctx.best_f1_scores_micro, \
            ctx.best_f1_scores_weighted = 0, 0, 0, 0, 0, 0
        ctx.last_acc, ctx.last_ari, ctx.last_f1_scores_median, ctx.last_f1_scores_macro, ctx.last_f1_scores_micro, \
            ctx.last_f1_scores_weighted = 0, 0, 0, 0, 0, 0
        test_loader = DataLoader(dataset=test_dataset_list[i], batch_size=batch_size, shuffle=True,
                                 pin_memory=False, collate_fn=ctx.pad_collate, drop_last=False)
        trainer.test_inner(test_loader, ctx)

        print(f'ctx acc list: {ctx.acc_list}')
        print(f'ctx auc list: {ctx.auc_list}')
        print(f'ctx ari list: {ctx.ari_list}')
        print(f'ctx f1_scores_median list: {ctx.f1_scores_median_list}')
        print(f'ctx f1_scores_macro list: {ctx.f1_scores_macro_list}')
        print(f'ctx f1_scores_micro list: {ctx.f1_scores_micro_list}')
        print(f'ctx f1_scores_weighted list: {ctx.f1_scores_weighted_list}')

        print(f'ctx best acc: {ctx.best_acc}')
        print(f'ctx best auc: {ctx.best_auc}')
        print(f'ctx best ari: {ctx.best_ari}')
        print(f'ctx best f1_scores_median: {ctx.best_f1_scores_median}')
        print(f'ctx best f1_scores_macro: {ctx.best_f1_scores_macro}')
        print(f'ctx best f1_scores_micro: {ctx.best_f1_scores_micro}')
        print(f'ctx best f1_scores_weighted: {ctx.best_f1_scores_weighted}')

        print(f'ctx last acc: {ctx.last_acc}')
        print(f'ctx last auc: {ctx.last_auc}')
        print(f'ctx last ari: {ctx.last_ari}')
        print(f'ctx last f1_scores_median: {ctx.last_f1_scores_median}')
        print(f'ctx last f1_scores_macro: {ctx.last_f1_scores_macro}')
        print(f'ctx last f1_scores_micro: {ctx.last_f1_scores_micro}')
        print(f'ctx last f1_scores_weighted: {ctx.last_f1_scores_weighted}')

        total_acc_list.append(ctx.acc_list.copy())
        total_ari_list.append(ctx.ari_list.copy())
        total_auc_list.append(ctx.auc_list.copy())
        total_f1_scores_median_list.append(ctx.f1_scores_median_list.copy())
        total_f1_scores_macro_list.append(ctx.f1_scores_macro_list.copy())
        total_f1_scores_micro_list.append(ctx.f1_scores_micro_list.copy())
        total_f1_scores_weighted_list.append(ctx.f1_scores_weighted_list.copy())

    ctx.acc_list = total_acc_list
    ctx.ari_list = total_ari_list
    ctx.auc_list = total_auc_list
    ctx.f1_scores_median_list = total_f1_scores_median_list
    ctx.f1_scores_macro_list = total_f1_scores_macro_list
    ctx.f1_scores_micro_list = total_f1_scores_micro_list
    ctx.f1_scores_weighted_list = total_f1_scores_weighted_list
    return ctx

def train_enhance_extra_with_d(word_dic_prefix, cell_type_prefix, train_file_path_list, test_file_path_list,
                               save_model_prefix, dir_name, times=1, print_postfix='',
                               print_prefix='', enhance_num=1,
                               execute_model='train'):
    if print_prefix is None:
        print_prefix = save_model_prefix

    auc_list = [[] for _ in range(len(test_file_path_list) + 1)]
    best_auc_list = [[] for _ in range(len(test_file_path_list) + 1)]
    last_auc_list = [[] for _ in range(len(test_file_path_list) + 1)]
    acc_list, ari_list, f1_scores_median_list, f1_scores_macro_list, f1_scores_micro_list, \
        f1_scores_weighted_list = [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in
                                                                                      range(
                                                                                          len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)]

    best_acc_list, best_ari_list, best_f1_scores_median_list, best_f1_scores_macro_list, best_f1_scores_micro_list, \
        best_f1_scores_weighted_list = [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in
                                                                                           range(
                                                                                               len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)]
    last_acc_list, last_ari_list, last_f1_scores_median_list, last_f1_scores_macro_list, last_f1_scores_micro_list, \
        last_f1_scores_weighted_list = [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in
                                                                                           range(
                                                                                               len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)]
    fine_tune_run_time_list, pretrain_run_time_list = [[] for _ in range(len(test_file_path_list) + 1)], \
        []
    torch.cuda.synchronize()
    start = time.time()
    train_enhance_contrastive_model(train_filepath=train_file_path_list,
                                    with_d=False,
                                    epoch=40,
                                    lr=0.0002,
                                    enhance_num=enhance_num,
                                    project_head_layer=None,
                                    mapping_file=[
                                        f'../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                                        f'../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
                                    save_model_path=f'pretrained/{save_model_prefix}_tissue_enhance1_1head_pretrained_cts_model.pth',
                                    d_model=64, h_dim=64, head=1, d_ffw=64 * 3, dropout_rate=0.2, vocab=40000,
                                    pca_num=64,
                                    batch_size=100,
                                    device_name='cuda:0', random_seed=0, continue_train=False)
    torch.cuda.synchronize()
    end = time.time()
    pretrain_run_time_list.append(end - start)
    for i in range(times):
        start = time.time()
        torch.cuda.synchronize()
        ctx = train_enhance_class_model_with_extra(train_filepath=train_file_path_list,
                                                   test_filepath=test_file_path_list, epoch=40,
                                                   freeze=True,
                                                   mlp_layer=[],
                                                   lr=0.001, enhance_num=enhance_num,
                                                   mapping_file=[
                                                       f'../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                                                       f'../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
                                                   save_model_path=f'pretrained/{save_model_prefix}_tissue_enhance1_1head_pretrained_class_model.pth',
                                                   trained_model_path=f'pretrained/{save_model_prefix}_tissue_enhance1_1head_pretrained_cts_model.pth',
                                                   d_model=64, h_dim=64, head=1, d_ffw=64 * 3, dropout_rate=0.2,
                                                   vocab=40000,
                                                   pca_num=64,
                                                   batch_size=100,
                                                   device_name='cuda:0', random_seed=i, continue_train=False)
        torch.cuda.synchronize()
        end = time.time()

        for j in range(len(test_file_path_list) + 1):
            fine_tune_run_time_list[j].append(end - start)
            acc_list[j] = ctx.acc_list[j]
            auc_list[j] = ctx.auc_list[j]
            ari_list[j] = ctx.ari_list[j]
            f1_scores_median_list[j] = ctx.f1_scores_median_list[j]
            f1_scores_macro_list[j] = ctx.f1_scores_macro_list[j]
            f1_scores_micro_list[j] = ctx.f1_scores_micro_list[j]
            f1_scores_weighted_list[j] = ctx.f1_scores_weighted_list[j]

            best_acc_list[j].append(max(acc_list[j]))
            best_auc_list[j].append(max(auc_list[j]))
            best_ari_list[j].append(max(ari_list[j]))
            best_f1_scores_median_list[j].append(max(f1_scores_median_list[j]))
            best_f1_scores_macro_list[j].append(max(f1_scores_macro_list[j]))
            best_f1_scores_micro_list[j].append(max(f1_scores_micro_list[j]))
            best_f1_scores_weighted_list[j].append(max(f1_scores_weighted_list[j]))

            last_acc_list[j].append(acc_list[j][-1])
            last_auc_list[j].append(auc_list[j][-1])
            last_ari_list[j].append(ari_list[j][-1])
            last_f1_scores_median_list[j].append(f1_scores_median_list[j][-1])
            last_f1_scores_macro_list[j].append(f1_scores_macro_list[j][-1])
            last_f1_scores_micro_list[j].append(f1_scores_micro_list[j][-1])
            last_f1_scores_weighted_list[j].append(f1_scores_weighted_list[j][-1])
        gc.collect()
    print('END TRAIN!!!!!!!!!!!!!!')
    import sys
    savedStdout = sys.stdout  # 保存标准输出流
    print_log = open(
        f"log/{print_prefix}_printlog{print_postfix}.txt", "a")
    sys.stdout = print_log
    for j in range(len(test_file_path_list) + 1):
        if j == 0:
            print(f'test file: total')
        else:
            continue
            print(f'test file: {test_file_path_list[j - 1]}')

        print(f'average pretrain run time: {np.mean(pretrain_run_time_list)}')
        print(f'pretrain run time list:{pretrain_run_time_list}')

        print(f'average fine tune run time: {np.mean(fine_tune_run_time_list[j])}')
        print(f'fine tune run time list:{fine_tune_run_time_list[j]}')

        print(f'average best acc: {np.mean(best_acc_list[j])}')
        print(f'best acc list:{best_acc_list[j]}')

        print(f'average best auc: {np.mean(best_auc_list[j])}')
        print(f'best auc list:{best_auc_list[j]}')

        print(f'average best ari: {np.mean(best_ari_list[j])}')
        print(f'best ari list:{best_ari_list[j]}')

        print(f'average best f1_scores_median: {np.mean(best_f1_scores_median_list[j])}')
        print(f'best f1_scores_median list:{best_f1_scores_median_list[j]}')

        print(f'average best f1_scores_macro: {np.mean(best_f1_scores_macro_list[j])}')
        print(f'best f1_scores_macro list:{best_f1_scores_macro_list[j]}')

        print(f'average best f1_scores_micro: {np.mean(best_f1_scores_micro_list[j])}')
        print(f'best f1_scores_micro list:{best_f1_scores_micro_list[j]}')

        print(f'average best f1_scores_weighted: {np.mean(best_f1_scores_weighted_list[j])}')
        print(f'best f1_scores_weighted list:{best_f1_scores_weighted_list[j]}')

        print(f'average last acc: {np.mean(last_acc_list[j])}')
        print(f'last acc list:{last_acc_list[j]}')

        print(f'average last auc: {np.mean(last_auc_list[j])}')
        print(f'last auc list:{last_auc_list[j]}')

        print(f'average last ari: {np.mean(last_ari_list[j])}')
        print(f'last ari list:{last_ari_list[j]}')

        print(f'average last f1_scores_median: {np.mean(last_f1_scores_median_list[j])}')
        print(f'last f1_scores_median list:{last_f1_scores_median_list[j]}')

        print(f'average last f1_scores_macro: {np.mean(last_f1_scores_macro_list[j])}')
        print(f'last f1_scores_macro list:{last_f1_scores_macro_list[j]}')

        print(f'average last f1_scores_micro: {np.mean(last_f1_scores_micro_list[j])}')
        print(f'last f1_scores_micro list:{last_f1_scores_micro_list[j]}')

        print(f'average last f1_scores_weighted: {np.mean(last_f1_scores_weighted_list[j])}')
        print(f'last f1_scores_weighted list:{last_f1_scores_weighted_list[j]}')
        print('\n')

    sys.stdout = savedStdout  # 恢复标准输出流
    print_log.close()
    gc.collect()

def train_enhance_extra_task2_with_d(word_dic_prefix, cell_type_prefix, train_file_path_list, test_file_path_list,
                                     save_model_prefix, dir_name, times=1, print_postfix='',
                                     print_prefix='', enhance_num=1,
                                     execute_model='train',test_data_name=''):
    if print_prefix is None:
        print_prefix = save_model_prefix

    auc_list = [[] for _ in range(len(test_file_path_list) + 1)]
    best_auc_list = [[] for _ in range(len(test_file_path_list) + 1)]
    last_auc_list = [[] for _ in range(len(test_file_path_list) + 1)]
    acc_list, ari_list, f1_scores_median_list, f1_scores_macro_list, f1_scores_micro_list, \
        f1_scores_weighted_list = [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in
                                                                                      range(
                                                                                          len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)]

    best_acc_list, best_ari_list, best_f1_scores_median_list, best_f1_scores_macro_list, best_f1_scores_micro_list, \
        best_f1_scores_weighted_list = [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in
                                                                                           range(
                                                                                               len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)]
    last_acc_list, last_ari_list, last_f1_scores_median_list, last_f1_scores_macro_list, last_f1_scores_micro_list, \
        last_f1_scores_weighted_list = [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in
                                                                                           range(
                                                                                               len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)], \
        [[] for _ in range(len(test_file_path_list) + 1)], [[] for _ in range(len(test_file_path_list) + 1)]
    fine_tune_run_time_list, pretrain_run_time_list = [[] for _ in range(len(test_file_path_list) + 1)], \
        []
    torch.cuda.synchronize()
    start = time.time()
    train_enhance_contrastive_model(train_filepath=train_file_path_list,
                                    # [1
                                    #                         '../../datasets/mouse/Testis/mouse_Testis_total.h5ad',
                                    #                   # ['../../datasets/mouse/Bladder/mouse_Bladder_total.h5ad']
                                    #                             '../../datasets/mouse/Brain/mouse_Brain_total.h5ad',
                                    #                             '../../datasets/mouse/Bladder/mouse_Bladder_total.h5ad'],
                                    with_d=False,
                                    epoch=40,
                                    lr=0.0002,
                                    enhance_num=enhance_num,
                                    project_head_layer=None,
                                    mapping_file=[
                                        f'../../datasets/preprocessed/{word_dic_prefix}_word_dic.pk',
                                        f'../../datasets/preprocessed/{cell_type_prefix}_cell_type_dic.pk'],
                                    save_model_path=f'pretrained/{save_model_prefix}_tissue_enhance1_1head_pretrained_cts_model_300_percent_pe_mean_with_tissue_without_D.pth',
                                    d_model=64, h_dim=64, head=1, d_ffw=64 * 3, dropout_rate=0.2, vocab=40000,
                                    pca_num=64,
                                    batch_size=100,
                                    device_name='cuda:0', random_seed=0, continue_train=False)
    torch.cuda.synchronize()
    end = time.time()
    pretrain_run_time_list.append(end - start)
    for i in range(times):
        start = time.time()
        torch.cuda.synchronize()
        ctx = train_enhance_class_model_with_extra(train_filepath=train_file_path_list,
                                                   test_filepath=test_file_path_list, epoch=40,
                                                   freeze=False,
                                                   mlp_layer=[],
                                                   lr=0.001, enhance_num=enhance_num,
                                                   mapping_file=[
                                                       f'../../datasets/preprocessed/{word_dic_prefix}_word_dic.pk',
                                                       f'../../datasets/preprocessed/{cell_type_prefix}_cell_type_dic.pk'],
                                                   save_model_path=f'pretrained/{save_model_prefix}_{i}_tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
                                                   trained_model_path=f'pretrained/{save_model_prefix}_tissue_enhance1_1head_pretrained_cts_model_300_percent_pe_mean_with_tissue_without_D.pth',
                                                   d_model=64, h_dim=64, head=1, d_ffw=64 * 3, dropout_rate=0.2,
                                                   vocab=40000,
                                                   pca_num=64,
                                                   batch_size=100,
                                                   device_name='cuda:0', random_seed=i, continue_train=False)
        torch.cuda.synchronize()
        end = time.time()

        for j in range(len(test_file_path_list) + 1):
            fine_tune_run_time_list[j].append(end - start)
            acc_list[j] = ctx.acc_list[j]
            auc_list[j] = ctx.auc_list[j]
            ari_list[j] = ctx.ari_list[j]
            f1_scores_median_list[j] = ctx.f1_scores_median_list[j]
            f1_scores_macro_list[j] = ctx.f1_scores_macro_list[j]
            f1_scores_micro_list[j] = ctx.f1_scores_micro_list[j]
            f1_scores_weighted_list[j] = ctx.f1_scores_weighted_list[j]

            best_acc_list[j].append(max(acc_list[j]))
            best_auc_list[j].append(max(auc_list[j]))
            best_ari_list[j].append(max(ari_list[j]))
            best_f1_scores_median_list[j].append(max(f1_scores_median_list[j]))
            best_f1_scores_macro_list[j].append(max(f1_scores_macro_list[j]))
            best_f1_scores_micro_list[j].append(max(f1_scores_micro_list[j]))
            best_f1_scores_weighted_list[j].append(max(f1_scores_weighted_list[j]))

            last_acc_list[j].append(acc_list[j][-1])
            last_auc_list[j].append(auc_list[j][-1])
            last_ari_list[j].append(ari_list[j][-1])
            last_f1_scores_median_list[j].append(f1_scores_median_list[j][-1])
            last_f1_scores_macro_list[j].append(f1_scores_macro_list[j][-1])
            last_f1_scores_micro_list[j].append(f1_scores_micro_list[j][-1])
            last_f1_scores_weighted_list[j].append(f1_scores_weighted_list[j][-1])
        gc.collect()
    print('END TRAIN!!!!!!!!!!!!!!')
    import sys
    savedStdout = sys.stdout  # 保存标准输出流
    # print_log = open(f"E:\论文\自己的论文\对比方法\scmodel\\ablation\extra2\{print_prefix}_printlog{print_postfix}.txt", "a")
    print_log = open(
        f"extra_result2/{print_prefix}_printlog{print_postfix}.txt", "a")
    sys.stdout = print_log
    for j in range(len(test_file_path_list) + 1):
        if j == 0:
            print(f'test file: total')
        else:
            break

        print(f'average pretrain run time: {np.mean(pretrain_run_time_list)}')
        print(f'pretrain run time list:{pretrain_run_time_list}')

        print(f'average fine tune run time: {np.mean(fine_tune_run_time_list[j])}')
        print(f'fine tune run time list:{fine_tune_run_time_list[j]}')

        print(f'average best acc: {np.mean(best_acc_list[j])}')
        print(f'best acc list:{best_acc_list[j]}')

        print(f'average best auc: {np.mean(best_auc_list[j])}')
        print(f'best auc list:{best_auc_list[j]}')

        print(f'average best ari: {np.mean(best_ari_list[j])}')
        print(f'best ari list:{best_ari_list[j]}')

        print(f'average best f1_scores_median: {np.mean(best_f1_scores_median_list[j])}')
        print(f'best f1_scores_median list:{best_f1_scores_median_list[j]}')

        print(f'average best f1_scores_macro: {np.mean(best_f1_scores_macro_list[j])}')
        print(f'best f1_scores_macro list:{best_f1_scores_macro_list[j]}')

        print(f'average best f1_scores_micro: {np.mean(best_f1_scores_micro_list[j])}')
        print(f'best f1_scores_micro list:{best_f1_scores_micro_list[j]}')

        print(f'average best f1_scores_weighted: {np.mean(best_f1_scores_weighted_list[j])}')
        print(f'best f1_scores_weighted list:{best_f1_scores_weighted_list[j]}')

        print(f'average last acc: {np.mean(last_acc_list[j])}')
        print(f'last acc list:{last_acc_list[j]}')

        print(f'average last auc: {np.mean(last_auc_list[j])}')
        print(f'last auc list:{last_auc_list[j]}')

        print(f'average last ari: {np.mean(last_ari_list[j])}')
        print(f'last ari list:{last_ari_list[j]}')

        print(f'average last f1_scores_median: {np.mean(last_f1_scores_median_list[j])}')
        print(f'last f1_scores_median list:{last_f1_scores_median_list[j]}')

        print(f'average last f1_scores_macro: {np.mean(last_f1_scores_macro_list[j])}')
        print(f'last f1_scores_macro list:{last_f1_scores_macro_list[j]}')

        print(f'average last f1_scores_micro: {np.mean(last_f1_scores_micro_list[j])}')
        print(f'last f1_scores_micro list:{last_f1_scores_micro_list[j]}')

        print(f'average last f1_scores_weighted: {np.mean(last_f1_scores_weighted_list[j])}')
        print(f'last f1_scores_weighted list:{last_f1_scores_weighted_list[j]}')
        print('\n')

    sys.stdout = savedStdout  # 恢复标准输出流
    print_log.close()
    gc.collect()

def show_enhance_embedding(dataset_name='Mouse-Pancreas-*', check_dataset_name='MCA-Pancreas', anndata_postfix='', i=0):
    # dataset_name = 'Mouse-Pancreas-*'
    visual_save_path_name = check_dataset_name + "_Extra_Task2" + "_TEST2"
    word_dic_prefix = check_dataset_name + "_Extra_Task2"
    cell_type_prefix = check_dataset_name + "_Extra_Task2"

    # visual_save_path_name = 'Baron'
    # visual_save_path_name = 'MCA-Pancreas'+ "_TEST"
    # visual_save_path_name = check_dataset_name
    # word_dic_prefix = 'MCA-Pancreas'
    # cell_type_prefix = 'MCA-Pancreas'

    # visual_save_path_name = 'PBMC45k-CEL-Seq2' + "_TEST"
    # word_dic_prefix = 'PBMC45k-CEL-Seq2'
    # cell_type_prefix = 'PBMC45k-CEL-Seq2'

    dir_name = 'cmp'
    adata_postfix = '.h5ad'
    data_files = glob.glob(f'../../../datasets/{dir_name}/{dataset_name}/*{adata_postfix}')

    print(list(data_files))
    f = list(data_files)
    # return
    # model_dataset_name = 'Romanov' + '_interpretable_'
    title_name = check_dataset_name
    model_dataset_name = check_dataset_name + "_Extra_Task2" +"_HVG2000"
    # model_dataset_name = check_dataset_name + "_Extra_Task2" + '_interpretable_'
    # show_embedding(
    #     # trained_model_path=f'pretrained/{model_dataset_name}_tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
    #     trained_model_path=f'pretrained/{model_dataset_name}_{i}_tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
    #     dataset_filepath=f, d_model=64, head=1, d_ffw=192, dropout_rate=0.2, enhance_num=1,
    #     mlp_layer=[],
    #     title_name=title_name,
    #     anndata_postfix=anndata_postfix+str(i),
    #     device_name="cuda:0",
    #     visual_save_path=visual_save_path_name,
    #     mapping_file=[f'../../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
    #                   f'../../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
    #     vocab=40000, batch_size=100)
    total_adata = None
    for ref_path in f:
        current_dataset_name, _ = os.path.splitext(os.path.basename(ref_path))
        if current_dataset_name == check_dataset_name:
            continue
        adata = check_anndata(ref_path)
        adata.var_names = [x.lower() for x in adata.var_names]
        adata.var_names_make_unique()
        print(adata.var_names)
        if total_adata is None:
            total_adata = adata
        else:
            total_adata = total_adata.concatenate(adata, join='outer', fill_value=0, uns_merge="first")
        del adata
        gc.collect()
    sc.pp.highly_variable_genes(total_adata, n_top_genes=2000, inplace=True, subset=True)
    hvg_names = total_adata.var_names.tolist()



def show_enhance_attention_weights(dataset_name='Mouse-Pancreas-MCA', check_dataset_name='MCA-Pancreas', k=10):
    # dataset_name = 'Mouse-Brain-Romanov'
    # word_dic_prefix = 'MCA-Pancreas' + "_Extra_Task2"
    # cell_type_prefix = 'MCA-Pancreas' + "_Extra_Task2"
    word_dic_prefix = check_dataset_name
    cell_type_prefix = check_dataset_name
    dir_name = 'cmp'
    adata_postfix = '.h5ad'
    data_files = glob.glob(f'../../../datasets/{dir_name}/{dataset_name}/*{adata_postfix}')
    print(list(data_files))
    f = list(data_files)
    # model_dataset_name = 'MCA-Pancreas' + "_Extra_Task2"
    model_dataset_name = check_dataset_name
    show_attention_weights(
        trained_model_path=f'pretrained/{model_dataset_name}_interpretable__tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
        dataset_filepath=f, d_model=64, h_dim=64, head=1, d_ffw=192, dropout_rate=0.2, enhance_num=1,
        mlp_layer=[],
        dataset_name=model_dataset_name + "_nozero",
        k=k,
        device_name="cuda:0",
        mapping_file=[f'../../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                      f'../../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
        vocab=40000, batch_size=100)

def train_extra_task1():
    dir_name = 'mouse'
    test_data = glob.glob('../../datasets/Mouse-Pancreas-*/*.h5ad')
    for j in range(len(test_data)):
        ref_data, _ = os.path.splitext(os.path.basename(test_data[j]))
        ref_file_path = test_data[j]
        tmp_test_data = []
        for i in range(len(test_data)):
            prefix, postfix = os.path.splitext(os.path.basename(test_data[i]))
            if prefix != ref_data:
                tmp_test_data.append(test_data[i])
        gc.collect()
        print(test_data)
        train_enhance_extra_with_d(dir_name=dir_name, word_dic_prefix=ref_data, cell_type_prefix=ref_data,
                                   train_file_path_list=[ref_file_path], test_file_path_list=tmp_test_data,
                                   enhance_num=1,
                                   save_model_prefix=ref_data, times=1,
                                   print_postfix='_pretrain40epoch_finetune40epoch_1layer_1head',
                                   print_prefix=ref_data)

    # test_data = glob.glob('../../../datasets/cmp/Mouse-Brain-*/*.h5ad')
    # for j in range(len(test_data)):
    #     # if j <= 1:
    #     #     continue
    #     ref_data, _ = os.path.splitext(os.path.basename(test_data[j]))
    #     # if ref_data != 'MCA-Pancreas':
    #     #     continue
    #     ref_file_path = test_data[j]
    #     tmp_test_data = []
    #     for i in range(len(test_data)):
    #         prefix, postfix = os.path.splitext(os.path.basename(test_data[i]))
    #         if prefix != ref_data:
    #             tmp_test_data.append(test_data[i])
    #     gc.collect()
    #     print(test_data)
    #     train_enhance_extra_with_d(dir_name=dir_name, word_dic_prefix=ref_data, cell_type_prefix=ref_data,
    #                                train_file_path_list=[ref_file_path], test_file_path_list=tmp_test_data,
    #                                enhance_num=1,
    #                                save_model_prefix=ref_data + '_interpretable' + "_nopca", times=5,
    #                                # save_model_prefix=ref_data, times=1,
    #                                print_postfix='_pretrain40epoch_finetune40epoch_nopca_1layer_1head_meanval_dot_embedding_nomaskenhance_embeddingdropout',
    #                                print_prefix=ref_data)

def train_extra_task2():
    dir_name = 'cmp'

    # total_data = glob.glob('../../../datasets/cmp/PBMC45kdonor-*/*.h5ad')
    # for j in range(len(total_data)):
    #     # if j <= 1:
    #     #     continue
    #     test_data, _ = os.path.splitext(os.path.basename(total_data[j]))
    #     test_file_path = total_data[j]
    #     tmp_ref_data = []
    #     for i in range(len(total_data)):
    #         prefix, postfix = os.path.splitext(os.path.basename(total_data[i]))
    #         if prefix != test_data:
    #             tmp_ref_data.append(total_data[i])
    #     gc.collect()
    #     print(f'check test data: {test_data}')
    #     train_enhance_extra_task2_with_d(dir_name=dir_name, word_dic_prefix=test_data + "_Extra_Task2",
    #                                      cell_type_prefix=test_data + "_Extra_Task2",
    #                                      train_file_path_list=tmp_ref_data, test_file_path_list=[test_file_path],
    #                                      enhance_num=1,
    #                                      save_model_prefix=test_data+ "_Extra_Task2", times=5,
    #                                      # save_model_prefix=test_data + "_Extra_Task2", times=5,
    #                                      # save_model_prefix=test_data + "_Extra_Task2" + '_big', times=1,
    #                                      print_postfix='_pretrain40epoch_finetune40epoch_1layer_1head_meanval_dot_embedding_nomaskenhance_noembeddingdropout',
    #                                      print_prefix=test_data + "_Extra_Task2",
    #                                      test_data_name=test_data)

    total_data = glob.glob('../../datasets/Mouse-Brain-*/*.h5ad')
    for j in range(len(total_data)):
        # if j <= 1:
        #     continue
        test_data, _ = os.path.splitext(os.path.basename(total_data[j]))
        # if test_data != 'TMS-Pancreas':
        #     continue
        test_file_path = total_data[j]
        tmp_ref_data = []
        for i in range(len(total_data)):
            prefix, postfix = os.path.splitext(os.path.basename(total_data[i]))
            if prefix != test_data:
                tmp_ref_data.append(total_data[i])
        gc.collect()
        print(f'check test data: {test_data}')
        train_enhance_extra_task2_with_d(dir_name=dir_name, word_dic_prefix=test_data + "_Extra_Task2",
                                         cell_type_prefix=test_data + "_Extra_Task2",
                                         train_file_path_list=tmp_ref_data, test_file_path_list=[test_file_path],
                                         enhance_num=1,
                                         save_model_prefix=test_data+"_Extra_Task2" + "_total", times=5,
                                         # save_model_prefix=test_data + "_Extra_Task2" + '_big', times=1,
                                         print_postfix='_pretrain40epoch_finetune40epoch_1layer_1head_val_dot_embedding_nomaskenhance_embeddingdropout',
                                         print_prefix=test_data + "_Extra_Task2_",
                                         test_data_name=test_data)


if __name__ == '__main__':
    # train_extra_task1()
    train_extra_task2()

    # show_enhance_attention_weights()
    # check_dataset_name_list = ['MCA-Pancreas', 'TMS-Pancreas', 'Baron']
    # # check_dataset_name_list = ['MCA-Pancreas']
    # for check_dataset_name in check_dataset_name_list:
    #     # show_enhance_embedding(dataset_name='Mouse-Pancreas-*', check_dataset_name=check_dataset_name)
    #     for i in range(5):
    #         show_enhance_embedding(dataset_name='Mouse-Pancreas-*', check_dataset_name=check_dataset_name, i=i)
    #         gc.collect()
    #
    # check_dataset_name_list = ['MCA-Brain', 'TMS-Brain', 'Romanov']
    # # check_dataset_name_list = ['TMS-Brain', 'Romanov']
    # for check_dataset_name in check_dataset_name_list:
    # #     show_enhance_embedding(dataset_name='Mouse-Brain-*', check_dataset_name=check_dataset_name)
    #     for i in range(5):
    #         show_enhance_embedding(dataset_name='Mouse-Brain-*', check_dataset_name=check_dataset_name, i=i)
    #         gc.collect()

    # dataset_name_list = ['Mouse-Pancreas-MCA', 'Mouse-Pancreas-TMS', 'Mouse-Pancreas-Baron',
    #                      'Mouse-Brain-MCA', 'Mouse-Brain-TMS', 'Mouse-Brain-Romanov']
    # dataname = ['MCA-Pancreas', 'TMS-Pancreas', 'Baron', 'MCA-Brain', 'TMS-Brain', 'Romanov']
    # for i in range(len(dataset_name_list)):
    #     show_enhance_attention_weights_prd(dataset_name=dataset_name_list[i], check_dataset_name=dataname[i], k=10)
        # show_enhance_attention_weights(dataset_name=dataset_name_list[i]+"_Extra_task2", check_dataset_name=dataname[i], k=0)

    # show_enhance_embedding()
    # test_data = glob.glob('../../../datasets/cmp/Mouse-Pancreas-*/*.h5ad')
    # print(test_data)
    # show_enhance_embedding()
    # generate_prd_label()
    # show_enhance_attention_weights()
    # dir_name='cmp'
    #
    # test_data = glob.glob('../../../datasets/cmp/PBMC45k-*/*.h5ad')
    # ref_data = 'PBMC45k-10x Chromium (v3)'
    # ref_file_path = 'test_data[j]'
    # c_idx = -1
    # for i in range(len(test_data)):
    #     prefix, postfix = os.path.splitext(os.path.basename(test_data[i]))
    #     if prefix == ref_data:
    #         c_idx = i
    #         ref_file_path = test_data[i]
    # if c_idx != -1:
    #     test_data.pop(c_idx)
    # print(test_data)
    # train_enhance_extra_with_d(dir_name=dir_name, word_dic_prefix=ref_data, cell_type_prefix=ref_data,
    #                            train_file_path_list=[ref_file_path],test_file_path_list=test_data,
    #                            save_model_prefix=ref_data, times=5,
    #                            print_postfix=ref_data+'with_threshold',
    #                            print_prefix=ref_data)

    # show_enhance_attention_weights()

    # dir_name = 'mouse'
    # dataset_name_list = ['Bladder', 'Bone_marrow', 'Brain', 'Embryonic_mesenchyme', 'Fetal_brain', 'Fetal_intestine', 'Fetal_liver', 'Fetal_lung', 'Fetal_stomach', 'Kidney', 'Liver', 'Lung', 'Mammary_gland', 'mouse_preprocess', 'mouse_total', 'Muscle', 'Neonatal_calvaria', 'Neonatal_heart', 'Neonatal_muscle', 'Neonatal_pancreas', 'Neonatal_rib', 'Neonatal_skin', 'Ovary', 'Pancreas', 'Peripheral_blood', 'Placenta', 'Prostate', 'Small_intestine', 'Spleen', 'Stomach', 'Testis', 'Thymus', 'Uterus']
    # dataset_name_list = ['Bladder', 'Embryonic_mesenchyme', 'Fetal_brain', 'Fetal_intestine',
    #                      'Fetal_liver', 'Fetal_lung', 'Fetal_stomach', 'Kidney', 'Liver', 'Lung',
    #                      'Mammary_gland', 'Neonatal_calvaria', 'Neonatal_heart',
    #                      'Neonatal_muscle', 'Neonatal_pancreas', 'Neonatal_rib', 'Neonatal_skin', 'Ovary', 'Pancreas',
    #                      'Placenta', 'Prostate', 'Small_intestine', 'Spleen', 'Thymus', 'Uterus',
    #                      'Muscle', 'Stomach', 'Peripheral_blood', 'Brain', 'Bone_marcd row', 'Testis']

    # dataset_name_list = ['Bladder', 'Embryonic_mesenchyme', 'Fetal_brain', 'Fetal_intestine',
    #                      'Fetal_liver', 'Fetal_lung', 'Fetal_stomach', 'Kidney', 'Liver', 'Lung',
    #                      'Mammary_gland', 'Neonatal_calvaria', 'Neonatal_heart',
    #                      'Neonatal_muscle', 'Neonatal_pancreas', 'Neonatal_rib', 'Neonatal_skin', 'Ovary', 'Pancreas',
    #                      'Placenta', 'Prostate', 'Small_intestine', 'Spleen', 'Thymus', 'Uterus']
    # dataset_name_list = ['Fetal_lung']
    # test_size_list = [0.99, 0.95, 0.9]
    # for dataset_name in dataset_name_list:
    #     train_enhance_with_d(test_size_list=test_size_list, dir_name=dir_name, dataset_name=dataset_name,
    #                          word_dic_prefix=dataset_name, cell_type_prefix=dataset_name, times=5,
    #                          print_postfix='_pretrain80epoch_noinitialpca_fintune80epoch', execute_model='train')

    # train_enhance_with_d(test_size_list=[0.9], dir_name='mouse', dataset_name='*', word_dic_prefix='mouse',
    #                      cell_type_prefix='mouse', times=3, print_prefix='mouse_total')
    # train_enhance_with_d(1)
    # show_enhance_attention_weights()
    # show_enhance_embedding()
    # test_enhance()
    # train(1)
