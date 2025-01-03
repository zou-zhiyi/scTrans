import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, csc_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import scanpy as sc

import heapq
from models.utils import WordIdxDic, check_anndata, merger_gene_dic, stratify_split, read_file_from_pickle


class SparsePredictDatasetPreprocessedV2(Dataset):
    def binary_generate(self, data, word_idx_dic: WordIdxDic, gene_name_list, left, right):
        middle = (left + right) // 2
        if left == right:
            print(f'current index: {middle}')
            # print(data[middle][:])
            # print(data[middle][:].A)
            # print(data.A.shape)
            data_val = data[0][:].A[0, :]
            # print(data_val.shape)
            data_mask = (data_val != 0)
            g_name_list = self.gene_name_list[data_mask]
            g_idx_list = np.array(list(
                map(lambda x: self.gene_idx_dic.getIdx(x.lower()) if self.gene_idx_dic.getIdx(x.lower()) else -1,
                    g_name_list)))

            g_idx_mask = (g_idx_list != -1)
            tissue_idx = None
            data_val = data_val[data_mask]
            data_val = data_val[g_idx_mask]
            g_idx_list = g_idx_list[g_idx_mask]

            # mean scaling
            # print(data_val)
            data_val = data_val / data_val.mean()
            # print(data_val)
            # print('=====')
            if tissue_idx is None:
                # print("tissue is None")
                # set cls embedding index
                tissue_idx = 1
            target_label = self.cell_type_idx_dic.getIdx(str(self.label[middle]).lower())
            if target_label is None:
                self.cell_type_idx_dic.insert(str(self.label[middle]).lower())
            target_label = self.cell_type_idx_dic.getIdx(str(self.label[middle]).lower())
            self.data_list.append((tissue_idx, g_idx_list, data_val, target_label))
        else:
            # print(current_left, current_right, current_middle)
            # print(left, right, middle)
            left_data = data[:middle + 1 - left]
            right_data = data[middle + 1 - left:]
            self.binary_generate(left_data, word_idx_dic, gene_name_list, left, middle)
            self.binary_generate(right_data, word_idx_dic, gene_name_list, middle+1, right)

    def __init__(self, data, label, word_idx_dic: WordIdxDic, cell_type_idx_dic: WordIdxDic, gene_name_list,
                 tissue=None):
        """
        将细胞类型映射单独拿出来,预测细胞类型使用另一个dic
        :param data:
        :param label:
        :param word_idx_dic:
        :param cell_type_idx_dic:
        :param gene_name_list:
        :param tissue:
        """
        # self.data = data
        self.gene_idx_dic = word_idx_dic
        self.cell_type_idx_dic = cell_type_idx_dic
        self.gene_name_list = gene_name_list
        self.tissue = tissue
        self.len = data.shape[0]
        self.label = label
        self.data_list = []
        self.label_dic = cell_type_idx_dic
        if type(data) == csr_matrix or type(data) == csc_matrix:
            self.binary_generate(data, word_idx_dic, gene_name_list, 0, self.len-1)
        else:
            for index in range(self.len):
                data_val = (data[index][:] != 0)
                g_name_list = self.gene_name_list[data_val]
                g_idx_list = np.array(list(
                    map(lambda x: self.gene_idx_dic.getIdx(x.lower()) if self.gene_idx_dic.getIdx(x.lower()) else -1,
                        g_name_list)))

                g_idx_mask = (g_idx_list != -1)
                tissue_idx = None
                data_val = data[index][data_val]
                data_val = data_val[g_idx_mask]
                g_idx_list = g_idx_list[g_idx_mask]

                # mean scaling
                # print(data_val)
                data_val = data_val / data_val.mean()
                # print(data_val)
                # print('=====')
                if tissue_idx is None:
                    # print("tissue is None")
                    tissue_idx = 1
                target_label = self.cell_type_idx_dic.getIdx(str(self.label[index]).lower())
                if target_label is None:
                    self.cell_type_idx_dic.insert(str(self.label[index]).lower())
                target_label = self.cell_type_idx_dic.getIdx(str(self.label[index]).lower())
                self.data_list.append((tissue_idx, g_idx_list, data_val, target_label))

    def __getitem__(self, index):
        tissue_idx, g_idx_list, data_val, target_label = self.data_list[index]
        tissue_idx = torch.tensor(tissue_idx)
        return tissue_idx, torch.ones_like(tissue_idx), torch.tensor(g_idx_list, dtype=torch.long) \
            , torch.tensor(data_val, dtype=torch.float32), torch.tensor(target_label)

    def __len__(self):
        return self.len


class SparsePredictDatasetPreprocessed_no_celltype(Dataset):
    def binary_generate(self, data, word_idx_dic: WordIdxDic, gene_name_list, left, right):
        middle = (left + right) // 2
        if left == right:
            print(f'current index: {middle}')
            # print(data[middle][:])
            # print(data[middle][:].A)
            # print(data.A.shape)
            data_val = data[0][:].A[0, :]
            # print(data_val.shape)
            data_mask = (data_val != 0)
            g_name_list = self.gene_name_list[data_mask]
            g_idx_list = np.array(list(
                map(lambda x: self.gene_idx_dic.getIdx(x.lower()) if self.gene_idx_dic.getIdx(x.lower()) else -1,
                    g_name_list)))

            g_idx_mask = (g_idx_list != -1)
            tissue_idx = None
            data_val = data_val[data_mask]
            data_val = data_val[g_idx_mask]
            g_idx_list = g_idx_list[g_idx_mask]

            # mean scaling
            # print(data_val)
            data_val = data_val / data_val.mean()
            # print(data_val)
            # print('=====')
            if tissue_idx is None:
                # print("tissue is None")
                # set cls embedding index
                tissue_idx = 1
            self.data_list.append((tissue_idx, g_idx_list, data_val))
        else:
            # print(current_left, current_right, current_middle)
            # print(left, right, middle)
            left_data = data[:middle + 1 - left]
            right_data = data[middle + 1 - left:]
            self.binary_generate(left_data, word_idx_dic, gene_name_list, left, middle)
            self.binary_generate(right_data, word_idx_dic, gene_name_list, middle+1, right)

    def __init__(self, data, word_idx_dic: WordIdxDic, gene_name_list,
                 tissue=None):
        """
        将细胞类型映射单独拿出来,预测细胞类型使用另一个dic
        :param data:
        :param label:
        :param word_idx_dic:
        :param cell_type_idx_dic:
        :param gene_name_list:
        :param tissue:
        """
        # self.data = data
        self.gene_idx_dic = word_idx_dic
        self.gene_name_list = np.array(gene_name_list)
        self.tissue = tissue
        self.len = data.shape[0]
        self.data_list = []
        if type(data) == csr_matrix or type(data) == csc_matrix:
            self.binary_generate(data, word_idx_dic, gene_name_list, 0, self.len-1)
        else:
            for index in range(self.len):
                # if index%100==0:
                #     print(index)

                    # data_val = data[index][:].A[0,:]
                    # data_mask = (data_val!=0)
                    # g_name_list = self.gene_name_list[data_mask]
                    #
                    # g_idx_list = np.array(list(
                    #     map(lambda x: self.gene_idx_dic.getIdx(x.lower()) if self.gene_idx_dic.getIdx(x.lower()) else -1,
                    #         g_name_list)))
                    #
                    # g_idx_mask = (g_idx_list != -1)
                    # tissue_idx = None
                    # data_val = data_val[data_mask]
                    # data_val = data_val[g_idx_mask]
                    # g_idx_list = g_idx_list[g_idx_mask]
                    #
                    # # mean scaling
                    # # print(data_val)
                    # data_val = data_val / data_val.mean()
                    # # print(data_val)
                    # # print('=====')
                    # if tissue_idx is None:
                    #     # print("tissue is None")
                    #     # set cls embedding index
                    #     tissue_idx = 1
                    # self.data_list.append((tissue_idx, g_idx_list, data_val))

                # else:
                    # if type(data) == np.ndarray:
                data_val = (data[index][:] != 0)
                g_name_list = self.gene_name_list[data_val]
                g_idx_list = np.array(list(
                    map(lambda x: self.gene_idx_dic.getIdx(x.lower()) if self.gene_idx_dic.getIdx(x.lower()) else -1,
                        g_name_list)))

                g_idx_mask = (g_idx_list != -1)
                tissue_idx = None
                data_val = data[index][data_val]
                data_val = data_val[g_idx_mask]
                g_idx_list = g_idx_list[g_idx_mask]

                # mean scaling
                # print(data_val)
                data_val = data_val / data_val.mean()
                # print(data_val)
                # print('=====')
                if tissue_idx is None:
                    # print("tissue is None")
                    # set cls embedding index
                    tissue_idx = 1
                self.data_list.append((tissue_idx, g_idx_list, data_val))

    def __getitem__(self, index):
        tissue_idx, g_idx_list, data_val = self.data_list[index]
        tissue_idx = torch.tensor(tissue_idx)
        return tissue_idx, torch.ones_like(tissue_idx), torch.tensor(g_idx_list, dtype=torch.long) \
            , torch.tensor(data_val, dtype=torch.float32)

    def __len__(self):
        return self.len


class SparsePredictDatasetPreprocessedV3_from_pk(Dataset):
    def __init__(self, pk_file_path):
        self.data_list = read_file_from_pickle(pk_file_path)
        self.len = len(self.data_list)

    def __getitem__(self, index):
        tissue_idx, g_idx_list, data_val, target_label = self.data_list[index]
        tissue_idx = torch.tensor(tissue_idx)
        return tissue_idx, torch.ones_like(tissue_idx), torch.tensor(g_idx_list, dtype=torch.long) \
            , torch.tensor(data_val, dtype=torch.float32), torch.tensor(target_label)

    def __len__(self):
        return self.len


class SparsePredictDatasetPreprocessedV3_from_list(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        self.len = len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return self.len


class PadCollate():
    def __init__(self):
        return

    def __call__(self, batch):
        # print('start pad')
        # print(batch)
        tissue_idx, tissue_val, feature_idx, feature_val, label = zip(*batch)
        feature_idx_lens = [len(x) for x in feature_idx]
        feature_val_lens = [len(y) for y in feature_val]
        feature_idx_pad = pad_sequence(feature_idx, batch_first=True, padding_value=0)
        feature_val_pad = pad_sequence(feature_val, batch_first=True, padding_value=0)
        feature_val_lens_tensor = torch.tensor(feature_val_lens)
        label_tensor = torch.tensor(label)
        del feature_val_lens
        del label
        return torch.tensor(tissue_idx), torch.tensor(tissue_val), feature_idx_pad, feature_val_pad, torch.tensor(
            feature_idx_lens), \
            feature_val_lens_tensor, label_tensor


class PadCollate_no_celltype():
    def __init__(self):
        return

    def __call__(self, batch):
        # print('start pad')
        # print(batch)
        tissue_idx, tissue_val, feature_idx, feature_val = zip(*batch)
        feature_idx_lens = [len(x) for x in feature_idx]
        feature_val_lens = [len(y) for y in feature_val]
        feature_idx_pad = pad_sequence(feature_idx, batch_first=True, padding_value=0)
        feature_val_pad = pad_sequence(feature_val, batch_first=True, padding_value=0)
        feature_val_lens_tensor = torch.tensor(feature_val_lens)
        del feature_val_lens
        return torch.tensor(tissue_idx), torch.tensor(tissue_val), feature_idx_pad, feature_val_pad, torch.tensor(
            feature_idx_lens), \
            feature_val_lens_tensor


class PadCollate2():
    def __init__(self):
        return

    def __call__(self, batch):
        # print('start pad')
        # print(batch)
        tissue_idx, tissue_val, feature_idx, feature_val, label, batch_id, raw_data = zip(*batch)
        feature_idx_lens = [len(x) for x in feature_idx]
        feature_val_lens = [len(y) for y in feature_val]
        feature_idx_pad = pad_sequence(feature_idx, batch_first=True, padding_value=0)
        feature_val_pad = pad_sequence(feature_val, batch_first=True, padding_value=0)
        feature_val_lens_tensor = torch.tensor(feature_val_lens)
        label_tensor = torch.tensor(label)
        batch_id_tensor = torch.tensor(batch_id)
        # print(f'batch id:{batch_id}')
        # print(f'raw data:{raw_data}')
        raw_data_tensor = torch.tensor(raw_data)
        del feature_val_lens
        del label
        return torch.tensor(tissue_idx), torch.tensor(tissue_val), feature_idx_pad, feature_val_pad, torch.tensor(
            feature_idx_lens), \
            feature_val_lens_tensor, label_tensor, batch_id_tensor, raw_data_tensor


class DenseData():
    def __init__(self, gene_idx_list, gene_val_list, rule_idx_list, rule_gene_idx_list):
        self.gene_idx_list = gene_idx_list
        self.gene_val_list = gene_val_list
        self.rule_idx_list = rule_idx_list
        self.rule_gene_idx_list = rule_gene_idx_list


def generate_train_test_dataset(filepath, test_size):
    adata = check_anndata(filepath)
    total_adata = check_anndata(filepath)
    word_idx_dic = WordIdxDic()
    word_idx_dic.insert('[pad]')
    word_idx_dic.insert('[cls]')
    cell_type_idx_dic = WordIdxDic()
    for cell_type in set(adata.obs['cell_type']):
        cell_type_idx_dic.insert(cell_type)
    # for filepath in filepath_list:
    word_idx_dic = merger_gene_dic(total_adata, word_idx_dic)
    total_dataset = SparsePredictDatasetPreprocessedV2(adata.X, adata.obs['cell_type'], word_idx_dic,
                                                       adata.var_names,
                                                       adata.obs["tissues"])
    train_dataset, test_dataset = stratify_split(total_data=total_dataset, test_size=test_size,
                                                 label_list=adata.obs['cell_type_idx'])
    return train_dataset, test_dataset, word_idx_dic, cell_type_idx_dic, adata


def generate_train_test_dataset_list(filepath_list, test_size, word_idx_dic, cell_type_idx_dic, random_seed):
    train_dataset_list, test_dataset_list, adata_list = [], [], []
    for filepath in filepath_list:
        adata = check_anndata(filepath)
        total_dataset = SparsePredictDatasetPreprocessedV2(adata.X, adata.obs['cell_type'], word_idx_dic,
                                                           cell_type_idx_dic,
                                                           adata.var_names,
                                                           adata.obs["batch_id"])
        train_dataset, test_dataset = stratify_split(total_data=total_dataset, test_size=test_size,
                                                     label_list=adata.obs['cell_type'],
                                                     random_seed=random_seed)
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
        adata_list.append(adata)
    return train_dataset_list, test_dataset_list, adata_list


def generate_train_test_dataset_list_from_pk(pk_filepath, label, test_size, random_seed):
    # print(adata.obs['cell_type'].value_counts())
    total_dataset = SparsePredictDatasetPreprocessedV3_from_pk(pk_filepath)
    # print(adata.obs['cell_type'])
    train_dataset, test_dataset = stratify_split(total_data=total_dataset, test_size=test_size,
                                                 label_list=label,
                                                 random_seed=random_seed)
    # test_dataset, _ = stratify_split(total_data=test_dataset, test_size=0.5,
    # label_list=label,
    # random_seed=random_seed)
    train_dataset = SparsePredictDatasetPreprocessedV3_from_list(train_dataset)
    test_dataset = SparsePredictDatasetPreprocessedV3_from_list(test_dataset)
    del total_dataset
    return train_dataset, test_dataset


def generate_dataset_list(filepath_list, word_idx_dic, cell_type_idx_dic):
    test_dataset_list, adata_list = [], []
    for filepath in filepath_list:
        adata = check_anndata(filepath)
        test_dataset = SparsePredictDatasetPreprocessedV2(adata.X, adata.obs['cell_type'], word_idx_dic,
                                                          cell_type_idx_dic,
                                                          adata.var_names)
        # test_dataset = SparsePredictDatasetPreprocessedV2(adata.X, adata.obs['cell_type'], word_idx_dic,
        #                                                   cell_type_idx_dic,
        #                                                   adata.var_names,
        #                                                   adata.obs["batch_id"])
        test_dataset_list.append(test_dataset)
        adata_list.append(adata)
    return test_dataset_list, adata_list


def generate_dataset_list_no_celltype(filepath_list, word_idx_dic):
    test_dataset_list, adata_list = [], []
    for filepath in filepath_list:
        adata = check_anndata(filepath)
        test_dataset = SparsePredictDatasetPreprocessed_no_celltype(adata.X, word_idx_dic,
                                                                    adata.var_names)
        # test_dataset = SparsePredictDatasetPreprocessedV2(adata.X, adata.obs['cell_type'], word_idx_dic,
        #                                                   cell_type_idx_dic,
        #                                                   adata.var_names,
        #                                                   adata.obs["batch_id"])
        test_dataset_list.append(test_dataset)
        adata_list.append(adata)
    return test_dataset_list, adata_list


def generate_dataset_list_with_hvg(filepath_list, word_idx_dic, cell_type_idx_dic, hvg_name_list):
    test_dataset_list, adata_list = [], []
    for filepath in filepath_list:
        adata = check_anndata(filepath)
        name_list = [x.lower() for x in adata.var_names]
        name_bool_index = np.in1d(name_list, hvg_name_list)
        print(np.any(name_bool_index))
        # print(len(hvg_name_list))
        test_dataset = SparsePredictDatasetPreprocessedV2(adata.X[:, name_bool_index],
                                                          adata.obs['cell_type'], word_idx_dic,
                                                          cell_type_idx_dic, np.array(name_list)[name_bool_index])
        test_dataset_list.append(test_dataset)
        new_adata = sc.AnnData(X=adata.X[:, name_bool_index])
        new_adata.var_names = np.array(name_list)[name_bool_index]
        adata_list.append(new_adata)
    return test_dataset_list, adata_list


def generate_dataset_from_pk(filepath):
    dataset = SparsePredictDatasetPreprocessedV3_from_pk(filepath)
    return dataset
