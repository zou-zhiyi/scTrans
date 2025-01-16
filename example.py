import glob

from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from models.utils import check_anndata, read_file_from_pickle, check_anndata_direct, calculate_score, set_seed
from scTrans_core import scTrans_core

import numpy as np
import scanpy as sc
import torch
import pandas as pd
import os


def pre_train_and_finetune(pre_train_adata, fine_tune_adata, save_path):

    sc_core = scTrans_core(file_save_path=save_path, device_name='cuda')
    sc_core.generate_mapping_file_for_embedding(gene_name_list=pre_train_adata.var_names.tolist())

    config = {
    'embedding_dim':64,
    'd_model':64,
    'h_dim':64,
    'head':1,
    'd_ffw':64 * 3,
    'dropout_rate':0.2,
    'vocab':60000,
    'embedding_dropout':False,
    'enhance_num':1,
    }
    sc_core.create_encoder(config)
    sc_core.embedding_initialize_from_pca([pre_train_adata])
    sc_core.pre_train([pre_train_adata], epoch=40)

    cell_type_list = fine_tune_adata.obs['cell_type'].unique().tolist()
    cell_type_number = len(cell_type_list)
    sc_core.generate_cell_type_mapping_file(cell_type_list=cell_type_list)
    config = {
        'd_model':64,
        'predict_type':cell_type_number,
        'dropout_rate':0.2
    }
    sc_core.create_classification_model(config)
    sc_core.fine_tune([fine_tune_adata], gene_freeze=True, epoch=40)

def predict(query_adata, cell_type_number, save_path):

    sc_core = scTrans_core(file_save_path=save_path, device_name='cuda')
    config = {
        'embedding_dim': 64,
        'd_model': 64,
        'h_dim': 64,
        'head': 1,
        'd_ffw': 64 * 3,
        'dropout_rate': 0.2,
        'vocab': 60000,
        'embedding_dropout': False,
        'enhance_num': 1,
    }
    sc_core.create_encoder(config)
    config = {
        'd_model': 64,
        'predict_type': cell_type_number,
        'dropout_rate': 0.2
    }
    sc_core.create_classification_model(config)

    sc_core.load_pretrained_model(saved_model_path=f'{save_path}/finetuned_model.pth', type='finetuned')

    sc_core.predict_cell_type(query_adata=query_adata, batch_size=100)

if __name__ == '__main__':
    adata = check_anndata('datasets/mouse_Muscle.h5ad')
    save_path = 'mouse_Muscle'
    pretrain_adata = adata
    index_list = [i for i in range(pretrain_adata.shape[0])]
    cell_type_list = adata.obs['cell_type'].tolist()
    train_index, test_index = train_test_split(index_list, test_size=0.9, train_size=0.1, stratify=cell_type_list)
    fine_tune_adata = adata[train_index, :].copy()
    query_adata = adata[test_index, :].copy()
    pre_train_and_finetune(pretrain_adata, fine_tune_adata, save_path)

    cell_type_list = fine_tune_adata.obs['cell_type'].unique().tolist()
    cell_type_number = len(cell_type_list)

    predict(query_adata=query_adata, cell_type_number=cell_type_number, save_path=save_path)
    cell_type_predict_list = query_adata.obs['cell_type_predict'].str.lower().tolist()
    cell_type_true_list = query_adata.obs['cell_type'].str.lower().tolist()
    acc, ari, f1_scores_median, f1_scores_macro, f1_scores_micro, f1_scores_weighted = calculate_score(cell_type_true_list, cell_type_predict_list)
    print(f'accuracy: {acc:.4f}, f1_macro: {f1_scores_macro:.4f}')
