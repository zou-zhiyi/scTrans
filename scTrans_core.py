import os

import scanpy as sc
import numpy as np
import torch
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, ConcatDataset

from models.dataset import PadCollate_no_celltype, SparsePredictDatasetPreprocessed_no_celltype, \
    SparsePredictDatasetPreprocessedV2, PadCollate
from models.impl.ClassificationTrainer import ClassificationTrainer
from models.impl.ContrastiveTrainer import ContrastiveTrainer
from models.model import Encoder, ClassificationModel
from models.train import Context
from models.utils import WordIdxDic, write_file_to_pickle, read_mapping_file, set_seed, read_file_from_pickle

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
            sample_idx = np.random.permutation(sample)
            gene_cell_matrix = gene_cell_matrix[:, sample_idx]
        # print(gene_cell_matrix.shape)
        gene_vec = pca.fit_transform(gene_cell_matrix)
        gene_names = adata.var_names
        gene_idx = gene_names.map(lambda x: gene_dic.getIdx(x.lower()))
        gene_idx = np.array(gene_idx).astype(int)
        print(f'gene idx: {gene_idx}')
        print(f'gene idx: {gene_idx[gene_idx >= 0]}')
        new_embedding[gene_idx[gene_idx >= 0]] = gene_vec[gene_idx >= 0]
    print(f'embedding initializing end')
    return new_embedding

class scTrans_core:

    def __init__(self, file_save_path, device_name):
        self.classification_model = None
        self.encoder_core = None
        self.device_name = device_name
        self.file_save_path = file_save_path
        self.gene_dic_save_path = f'{file_save_path}/reference_word_dic.pk'
        self.cell_type_dic_save_path = f'{file_save_path}/reference_cell_type_dic.pk'
        self.pretrained_save_model_path = f'{file_save_path}/pretrained_model.pth'
        self.fine_tuned_save_model_path = f'{file_save_path}/finetuned_model.pth'

        if os.path.exists(self.file_save_path) == False:
            os.makedirs(self.file_save_path)

    def load_pretrained_model(self, saved_model_path, type='finetuned'):
        if type == 'finetuned':
            self.classification_model.load_state_dict(torch.load(saved_model_path)['model'])
            self.encoder_core = self.classification_model.encode
        elif type == 'pretrained':
            self.encoder_core.load_state_dict(torch.load(saved_model_path)['model'])
        else:
            print('not find type')

    def generate_mapping_file_for_embedding(self, gene_name_list):
        gene_idx_dic = WordIdxDic()
        gene_idx_dic.insert('[pad]')
        gene_idx_dic.insert('[cls]')
        for gene in gene_name_list:
            gene_idx_dic.insert(gene.lower())
        write_file_to_pickle(gene_idx_dic, self.gene_dic_save_path)
        return gene_idx_dic

    def generate_cell_type_mapping_file(self, cell_type_list):
        cell_type_idx_dic = WordIdxDic()
        for ct in cell_type_list:
            cell_type_idx_dic.insert(str(ct).lower())
        write_file_to_pickle(cell_type_idx_dic, self.cell_type_dic_save_path)
        return cell_type_idx_dic

    def create_encoder(self, config:{}):
        self.embedding_dim = config['embedding_dim']
        d_model = config['d_model']
        h_dim = config['h_dim']
        head = config['head']
        d_ffw = config['d_ffw']
        dropout_rate = config['dropout_rate']
        self.vocab = config['vocab']
        embedding_dropout = config['embedding_dropout']
        enhance_num = config['enhance_num']
        self.encoder_core =  Encoder(embedding_dim=self.embedding_dim, d_model=d_model, h_dim=h_dim, head=head, d_ffw=d_ffw,
                         dropout_rate=dropout_rate,
                         vocab=self.vocab, embedding_dropout=embedding_dropout,
                         enhance_num=enhance_num)

    def create_classification_model(self, config:{}):
        d_model = config['d_model']
        predict_type = config['predict_type']
        dropout_rate = config['dropout_rate']
        self.classification_model = ClassificationModel(model=self.encoder_core,
                                                        d_model=d_model, predict_type=predict_type,
                                                        dropout_rate=dropout_rate)

    def embedding_initialize_from_pca(self, adata_list, sample=None):
        embedding = embedding_pca_initializing(adata_list, self.embedding_dim, self.gene_dic_save_path, self.vocab, sample=sample)
        self.embedding_initialize(embedding)

    def embedding_initialize(self, embedding):
        self.encoder_core.embedding.feature_embedding.word_embed = \
            torch.nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float32,
                                                            device=torch.device(self.device_name)), freeze=False,
                                                            padding_idx=0)

    def pre_train(self, adata_list, epoch=40, batch_size=100, lr=0.0002, gene_freeze=False, random_seed=None):
        set_seed(random_seed)
        if gene_freeze:
            self.encoder_core.embedding.requires_grad_(False)
        # if cls_freeze:
        #     self.encoder_core.cls_embedding.requires_grad_(False)
        gene_dic = read_file_from_pickle(self.gene_dic_save_path)
        total_train_dataset = []
        for adata in adata_list:
            train_dataset = SparsePredictDatasetPreprocessed_no_celltype(adata.X, gene_dic,
                                                                        adata.var_names)
            total_train_dataset.append(train_dataset)
        total_train_dataset = ConcatDataset(total_train_dataset)

        trainer = ContrastiveTrainer(self.encoder_core, [total_train_dataset], [], device_name=self.device_name, lr=lr,
                                     save_path=self.file_save_path)
        ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=self.pretrained_save_model_path,
                      pad_collate=PadCollate_no_celltype(), random_seed=None)
        trainer.train(ctx)
        pass


    def fine_tune(self, reference_adata_list, lr=0.001, epoch=40, batch_size=100, gene_freeze=False, random_seed=None):
        if gene_freeze:
            self.encoder_core.embedding.requires_grad_(False)
        # if cls_freeze:
        #     self.encoder_core.cls_embedding.requires_grad_(False)

        gene_dic = read_file_from_pickle(self.gene_dic_save_path)
        cell_type_idx_dic =  read_file_from_pickle(self.cell_type_dic_save_path)

        total_train_dataset = []
        for ref in reference_adata_list:
            train_dataset = SparsePredictDatasetPreprocessedV2(ref.X, ref.obs['cell_type'],
                                                               gene_dic,
                                                               cell_type_idx_dic,
                                                               ref.var_names)
            total_train_dataset.append(train_dataset)
        total_train_dataset = ConcatDataset(total_train_dataset)

        trainer = ClassificationTrainer(self.classification_model, [total_train_dataset], test_dataset=None,
                                        trained_model_path=None, device_name=self.device_name, lr=lr,
                                        save_path=self.file_save_path)
        ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=self.fine_tuned_save_model_path,
                      pad_collate=PadCollate(), random_seed=None)
        trainer.train(ctx)

    def predict_cell_type(self, query_adata:sc.AnnData, batch_size=100):
        gene_dic = read_file_from_pickle(self.gene_dic_save_path)
        cell_type_idx_dic =  read_file_from_pickle(self.cell_type_dic_save_path)
        target_dataset = SparsePredictDatasetPreprocessed_no_celltype(query_adata.X, gene_dic,
                                                                    query_adata.var_names)
        trainer = ClassificationTrainer(self.classification_model, None, test_dataset=None,
                                        trained_model_path=None, device_name=self.device_name,
                                        save_path=self.file_save_path)
        ctx = Context(batch_size=batch_size, save_model_path=self.fine_tuned_save_model_path)
        data_loader = DataLoader(target_dataset,batch_size=batch_size, shuffle=False,
                                       collate_fn=PadCollate_no_celltype())
        ctx.data_loader = data_loader
        ctx.cell_type_idx_dic = cell_type_idx_dic
        trainer.predict_cell_type(ctx)
        predict_cell_type_list = ctx.cell_type_prd_list
        query_adata.obs['cell_type_predict'] = predict_cell_type_list

    def generate_embedding(self, query_adata:sc.AnnData, batch_size=100):
        gene_dic = read_file_from_pickle(self.gene_dic_save_path)
        train_dataset = SparsePredictDatasetPreprocessed_no_celltype(query_adata.X, gene_dic,
                                                                    query_adata.var_names)
        data_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False,
                                       collate_fn=PadCollate_no_celltype())

        trainer = ContrastiveTrainer(self.encoder_core, [], [], device_name=self.device_name,
                                     save_path=self.file_save_path)
        ctx = Context(batch_size=batch_size)
        ctx.data_loader = data_loader
        trainer.generate_new_embedding(ctx)
        embedding = ctx.n_embedding

        query_adata.obsm['embedding'] = embedding

    def generate_attetion_weights(self, query_adata:sc.AnnData, batch_size=100):
        gene_dic = read_file_from_pickle(self.gene_dic_save_path)
        train_dataset = SparsePredictDatasetPreprocessed_no_celltype(query_adata.X, gene_dic,
                                                                    query_adata.var_names)
        data_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=False,
                                       collate_fn=PadCollate_no_celltype())

        trainer = ContrastiveTrainer(self.encoder_core, [], [], device_name=self.device_name,
                                     save_path=self.file_save_path)
        ctx = Context(batch_size=batch_size)
        ctx.data_loader = data_loader
        ctx.gene_num = len(gene_dic.word2idx_dic)
        trainer.show_attention_weights(ctx)
        n_attention_weights = ctx.n_attention_weights

        attention_weights_adata = sc.AnnData(X=n_attention_weights)
        id_list = [i for i in range(n_attention_weights.shape[-1])]
        gene_names = list(map(lambda x:gene_dic.getGene(x), id_list))
        attention_weights_adata.obs_names = query_adata.obs_names.tolist()
        attention_weights_adata.var_names = gene_names
        return attention_weights_adata