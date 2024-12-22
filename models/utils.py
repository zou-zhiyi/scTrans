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

def read_data_from_csv(data_path, cell_type_path, dataset_prefix, check_data=False):
    rna_data = pd.read_csv(data_path, header=None, low_memory=False)
    cell_type_data = pd.read_csv(cell_type_path, header=None, low_memory=False)
    adata = sc.AnnData(np.array(rna_data.iloc[1:, 1:].transpose(), dtype=np.float32))
    # print(adata.shape)
    # print(rna_data.iloc[1:, 0])
    # print(rna_data.iloc[1:, 0].shape)
    adata.var['gene_name'] = np.array(rna_data.iloc[1:, 0])
    adata.var_names = np.array(rna_data.iloc[1:, 0])
    adata.obs_names = np.array(rna_data.iloc[0, 1:])
    print(dataset_prefix)
    adata.obs['cell_name'] = np.array(rna_data.iloc[0, 1:] + dataset_prefix)
    # print(cell_type_data.iloc[:, 2])
    adata.obs['cell_type'] = np.array(cell_type_data.iloc[1:, 2])

    if check_data:
        check_anndata_direct(adata)
    return adata
    # cell_type_data = pd.read_csv(cell_type_path, header=None, low_memory=False)
    # check_anndata_direct(adata)
    # write_to_h5ad(adata, f'Bone_h5/{data_path}')



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


def merge_files(file_prefixes, save_filename, tissue_name):
    cell_nums = 0
    gene_set = set()
    cell_type_set = set()
    total_adata = None
    if len(file_prefixes) == 0:
        return
    # print(data_files)
    for file_prefix in file_prefixes:
        print(file_prefix)
        data_files = glob.glob(f'{file_prefix}*_data.csv')
        cnt = 0
        for data_file in data_files:
            father_path = os.path.abspath((os.path.dirname(data_file)))
            # print(father_path)
            # print(data_file)

            cell_type_file = data_file.split(os.sep)[-1].split('_')
            cell_type_file[-1] = 'celltype.csv'
            cell_type_file = '_'.join(cell_type_file)
            print(f'{cell_type_file}')
            adata = read_data_from_csv(data_file, f'{father_path}{os.sep}{cell_type_file}', '_' + cell_type_file, True)
            cell_nums += len(adata.obs)
            gene_set.update(set(np.array(adata.var['gene_name'])))
            cell_type_set.update(set(np.array(adata.obs['cell_type'])))
            adata.obs['tissues'] = tissue_name
            adata.obs['batch_id'] = cnt
            cnt += 1
            if total_adata is None:
                total_adata = adata
            else:
                total_adata = total_adata.concatenate(adata, join='outer', fill_value=0, uns_merge="first")
            gc.collect()
        print(f'cell nums: {cell_nums}')
        print(f'gene set nums: {len(gene_set)}')
        print(f'cell type set nums: {len(cell_type_set)}')
        print(cell_type_set)
        if total_adata.uns.get('tissues', None) is None:
            total_adata.uns['tissues'] = [tissue_name]
        else:
            total_adata.uns['tissues'].append(tissue_name)
    sc.pp.highly_variable_genes(total_adata)
    label_transform(total_adata, save_filename)
    check_anndata_direct(total_adata)
    print(total_adata.obs["cell_type_idx"])
    print(set(total_adata.obs["cell_type_idx"]))
    return total_adata

def merge_multi_files(fileprefx, save_filename, tissue_name=None):
    if isinstance(fileprefx, list):
        return merge_files(fileprefx, save_filename, tissue_name)
    else:
        return merge_files([fileprefx], save_filename, tissue_name)

def loss_visual(loss_total, test_loss_total, idx=''):
    plt.cla()
    plt.clf()
    y = loss_total
    print(loss_total)
    if not os.path.exists('loss'):
        os.mkdir('loss')
    x = [i for i in range(len(y))]
    plt.plot(x, y)
    x = [i for i in range(len(test_loss_total))]
    plt.plot(x, test_loss_total)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    if not os.path.exists('loss'):
        os.mkdir('loss')
    plt.savefig('loss/loss_' + str(idx) + '_' + datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S") + '.jpg')
    # plt.show()
    print('plt saved')
    plt.close()


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = "#" + ''.join([random.choice(colorArr) for i in range(6)])
    return color


def randommarker():
    marker_list = ['o', 'v', '^', '<', '>', 's', 'p', '*', 'D', 'P', 'X']
    marker = random.choice(marker_list)
    return marker


def umap_plot(data, label_name, save_file_name):
    plt.figure(figsize=(10, 10), dpi=300)
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)
    # nan_error_value = adata.obs['cell_type_idx'].min()
    # max_type_value = adata.obs['cell_type_idx'].max()
    # adata.obs['cell_type_idx'][adata.obs['cell_type_idx'] == nan_error_value] = 0
    # target = adata.obs['cell_type_idx']
    # plt.subplots(300)
    # print(embbeding.shape)
    label_set = set(label_name.tolist())
    cnt = 0
    for l in label_set:
        tmp = (label_name == l)
        # print(tmp.shape)
        # print(tmp.sum())
        plt.scatter(embedding[tmp, 0], embedding[tmp, 1], marker='o', c=randomcolor(), s=5, label=l)
        cnt += 1
    plt.legend(loc="upper right", title="Classes")

    # legend = ax.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
    # ax.add_artist(legend)
    plt.savefig(save_file_name)
    plt.show()
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
