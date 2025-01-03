import datetime
import glob
import random
import seaborn as sns
import pickle
import os
from collections import Counter

import sklearn.metrics
from sklearn.cluster import KMeans
from typing import List

import numpy as np
import torch
import scanpy as sc
import anndata as ad
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, accuracy_score, f1_score, silhouette_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn import manifold
from torch import nn
import copy
import pandas as pd
import gc
import ast


class WordIdxDic:

    def __init__(self):
        self.word2idx_dic = {}
        self.idx2word_dic = {}
        self.current_idx = 0

    def insert(self, gene):
        if gene in self.word2idx_dic.keys():
            return
        else:
            while self.current_idx in self.idx2word_dic.keys():
                self.current_idx += 1
            self.word2idx_dic[gene] = self.current_idx
            self.idx2word_dic[self.current_idx] = gene

    def getGene(self, idx):
        return self.idx2word_dic.get(idx, None)

    def getIdx(self, gene):
        return self.word2idx_dic.get(gene, None)


def write_file_to_pickle(data, save_path):
    with open(save_path, 'wb') as file_to_write:
        pickle.dump(data, file_to_write)


def read_file_from_pickle(save_path):
    if save_path is None:
        return None
    with open(save_path, 'rb') as file_to_read:
        data = pickle.load(file_to_read)
    return data


def merger_gene_dic(adata: sc.AnnData, gene_idx_dic=None) -> WordIdxDic:
    if gene_idx_dic is None:
        gene_idx_dic = WordIdxDic()
    for gene in adata.var_names:
        gene_idx_dic.insert(gene.lower())
    if 'cell_type' in adata.obs.keys():
        for cell_type in set(adata.obs['cell_type']):
            gene_idx_dic.insert(str(cell_type).lower())
    if adata.uns.get('tissues', None) is not None:
        for tissue in adata.uns['tissues']:
            gene_idx_dic.insert(tissue.lower())
    return gene_idx_dic


def merger_gene_dic_from_varname(var_names, cell_types, gene_idx_dic=None) -> WordIdxDic:
    if gene_idx_dic is None:
        gene_idx_dic = WordIdxDic()
    for gene in var_names:
        if isinstance(gene, str):
            gene_idx_dic.insert(gene.lower())
        else:
            gene_idx_dic.insert(gene)
    for cell_type in set(cell_types):
        if isinstance(cell_type, str):
            gene_idx_dic.insert(cell_type.lower())
        else:
            gene_idx_dic.insert(cell_type)
    # if adata.uns.get('tissues', None) is not None:
    #     for tissue in adata.uns['tissues']:
    #         gene_idx_dic.insert(tissue.lower())
    return gene_idx_dic



def label_transform(adata: sc.AnnData, filepath):
    cell_type_set = set(adata.obs['cell_type'])
    cell_type_dic = {}
    cnt = 0
    for ct in cell_type_set:
        cell_type_dic[ct] = cnt
        cnt += 1
    adata.obs['cell_type_idx'] = adata.obs['cell_type'].map(lambda x: cell_type_dic[x])
    # adata.uns['cell_type_nums'] = cnt
    # adata.uns['cell_type_dic'] = cell_type_dic
    check_anndata_direct(adata)
    print(f"start to write: {filepath}")
    print(f'before gene nums:{len(adata.var_names)}')
    sc.pp.filter_genes(adata, min_cells=1)
    print(f'after gene nums:{len(adata.var_names)}')
    write_to_h5ad(adata, filepath)



def loss_visual(loss_total, test_loss_total,save_path='loss', idx=''):
    plt.cla()
    plt.clf()
    y = loss_total
    print(loss_total)
    x = [i for i in range(len(y))]
    plt.plot(x, y)
    x = [i for i in range(len(test_loss_total))]
    plt.plot(x, test_loss_total)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    if not os.path.exists(f'{save_path}/loss'):
        os.makedirs(f'{save_path}/loss')
    plt.savefig(f'{save_path}/loss/loss_' + str(idx) + '_' + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + '.jpg')
    # plt.show()
    print('plt saved')
    plt.close()


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def generate_mapping_file(filepath_list: List[str], word_save_path, cell_type_save_path, key='cell_type'):
    word_idx_dic = WordIdxDic()
    word_idx_dic.insert('[pad]')
    word_idx_dic.insert('[cls]')
    cell_type_idx_dic = WordIdxDic()
    for filepath in filepath_list:
        adata = check_anndata(filepath)
        word_idx_dic = merger_gene_dic(adata, word_idx_dic)
        if key in adata.obs.keys():
            for cell_type in set(adata.obs[key]):
                cell_type_idx_dic.insert(str(cell_type).lower())
    write_file_to_pickle(word_idx_dic, word_save_path)
    write_file_to_pickle(cell_type_idx_dic, cell_type_save_path)

def read_mapping_file(word_save_path, cell_type_save_path):
    word_idx_idc, cell_type_idx_dic = \
        read_file_from_pickle(word_save_path), read_file_from_pickle(cell_type_save_path)
    return word_idx_idc, cell_type_idx_dic


def check_anndata_direct(data):
    print("Data matrix:")
    print(data.shape)
    print(data.X)
    print("=======================")
    print("Data obs:")
    print(data.obs)
    print("=======================")
    print("Data obs keys")
    print(data.obs.keys())
    print("=======================")
    print("Data var:")
    print(data.var)
    print("=======================")
    print("Data var keys")
    print(data.var.keys())
    print("=======================")
    print("Data uns:")
    print(data.uns)
    print("=======================")


def check_anndata(filepath, is_print=False):
    data = ad.read_h5ad(filepath)
    if is_print:
        print("Data matrix:")
        print(data.shape)
        print(data.X)
        print("=======================")
        print("Data obs:")
        print(data.obs)
        print("=======================")
        print("Data obs keys")
        print(data.obs.keys())
        print("=======================")
        print("Data var:")
        print(data.var)
        print("=======================")
        print("Data var keys")
        print(data.var.keys())
        print("=======================")
        print("Data uns:")
        print(data.uns)
        print("=======================")
    return data


def write_to_h5ad(anndata, filepath, copy=False):
    anndata.write_h5ad(filepath)
    if copy:
        anndata.write_h5ad(filepath + "_copy")


def stratify_split(total_data, random_seed=None, test_size=0.1, label_list=None):
    # print(Counter(total_data.obs['Cluster']))
    if label_list is not None:
        train_data, test_data = train_test_split(total_data, stratify=label_list, test_size=test_size,
                                                 random_state=random_seed)
    else:
        train_data, test_data = train_test_split(total_data, test_size=test_size, random_state=random_seed)
    # print(Counter(test_data.obs['Cluster']))
    return train_data, test_data


def calculate_score(true_label, pred_label):
    ari = adjusted_rand_score(true_label, pred_label)
    acc = accuracy_score(true_label, pred_label)
    # acc = precision_score(true_label,pred_label,average='macro')
    f1_scores_median = f1_score(true_label, pred_label, average=None)
    # print(f'f1 list: {f1_scores_median}')
    f1_scores_median = np.median(f1_scores_median)
    f1_scores_macro = f1_score(true_label, pred_label, average='macro')
    f1_scores_micro = f1_score(true_label, pred_label, average='micro')
    f1_scores_weighted = f1_score(true_label, pred_label, average='weighted')
    # print('acc:', acc, 'ari:', ari, 'f1_scores_median:', f1_scores_median, 'f1_scores_macro:',
    #       f1_scores_macro, 'f1_scores_micro:', f1_scores_micro, 'f1_scores_weighted:', f1_scores_weighted)
    return acc, ari, f1_scores_median, f1_scores_macro, f1_scores_micro, f1_scores_weighted


def save_model(model, opt, model_path):
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, model_path)


def set_seed(seed=None):
    if seed is not None:
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
        torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
        np.random.seed(seed)  # numpy产生的随机数一致
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    # torch.backends.cudnn.deterministic = True

    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    # torch.backends.cudnn.benchmark = False
