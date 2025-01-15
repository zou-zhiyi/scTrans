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

def embedding_pca_initializing(adata_list, pca_num, gene_idx_idc_path, vocab, sample=None):
    gene_dic = read_file_from_pickle(gene_idx_idc_path)
    pca = PCA(n_components=pca_num)
    new_embedding = np.zeros((vocab, pca_num))
    for adata in adata_list:
        if type(adata.X) == csr_matrix or type(adata.X) == csc_matrix:
            gene_cell_matrix = adata.X.toarray().T
        else:
            gene_cell_matrix = np.array(adata.X).T
        if sample is not None:
            gene_cell_matrix = gene_cell_matrix[:, :sample]
            # sample_idx = np.random.permutation(sample)
        gene_vec = pca.fit_transform(gene_cell_matrix)
        gene_names = adata.var_names
        gene_idx = gene_names.map(lambda x: gene_dic.getIdx(x.lower()))
        gene_idx = np.array(gene_idx).astype(int)
        print(f'gene idx: {gene_idx}')
        print(f'gene idx: {gene_idx[gene_idx >= 0]}')
        new_embedding[gene_idx[gene_idx >= 0]] = gene_vec[gene_idx >= 0]
    print(f'embedding initializing end')
    return new_embedding

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
    embedding = embedding_pca_initializing([pre_train_adata], config['embedding_dim'], sc_core.gene_dic_save_path,config['vocab'],
                                           sample=30000)
    sc_core.embedding_initialize(embedding)
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
    sc_core.fine_tune([fine_tune_adata], freeze=True, epoch=40)

def predict(query_adata, fine_tune_adata, save_path):
    cell_type_list = fine_tune_adata.obs['cell_type'].unique().tolist()
    cell_type_number = len(cell_type_list)

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
    predict_results_df = pd.DataFrame({'cell_type': query_adata.obs['cell_type'].tolist(), 'cell_type_predict': query_adata.obs['cell_type_predict'].tolist()})
    predict_results_df.to_csv(f'{save_path}/results.csv', index=True)
