import gc

import glob
import time
from collections import Counter

import sklearn.metrics.pairwise
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from models.dataset import generate_train_test_dataset_list, PadCollate, \
    generate_dataset_list, generate_train_test_dataset_list_from_pk
from models.impl.ContrastiveTrainer import train_enhance_contrastive, train_enhance_contrastive_from_pk
from models.model import LabelSmoothing, generate_enhance_classification_model_with_d
from models.train import Trainer, Context, enhance_classification_construct

from models.utils import set_seed, calculate_score, write_file_to_pickle, check_anndata, read_mapping_file

import pandas as pd


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

    def train_inner(self, train_loader, context: Optional[Context]):
        cross_entropy_loss = LabelSmoothing(0)
        loss_sum = []
        y_true_list = []
        for i, batch in enumerate(train_loader):
            # print(batch)
            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad, feature_idx_lens, \
                feature_val_lens, label = batch
            current_batch_size = tissue_idx.shape[0]

            # batch size 为1时， batch norm会出现问题
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

            # 预测结果，由于CLStoken对应的index都为1，所以直接使用torch.ones_like生成CLS index
            prd = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad, feature_val_pad,
                             key_padding_mask, None)

            # 交叉熵损失
            softmax_loss = cross_entropy_loss(prd, label)

            self.opt.zero_grad()
            softmax_loss.backward()
            self.opt.step()
            loss_sum.append(softmax_loss.item())
        context.epoch_loss = np.mean(loss_sum)

    def test_inner(self, test_loader, context: Optional[Context]):
        cross_entropy_loss = LabelSmoothing(0)
        loss_sum = []
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

            tissue_idx, tissue_val, feature_idx_pad, feature_val_pad = tissue_idx.to(self.device), \
                tissue_val.to(self.device), feature_idx_pad.to(self.device), feature_val_pad.to(self.device)
            label = label.to(self.device).unsqueeze(-1)

            prd = self.model(torch.ones_like(tissue_idx), tissue_val, feature_idx_pad, feature_val_pad,
                             key_padding_mask, None)

            softmax_loss = cross_entropy_loss(prd, label)

            prd = torch.softmax(prd, dim=-1)

            if y_prop_list is None:
                y_prop_list = prd.detach().cpu().numpy()
            else:
                y_prop_list = np.concatenate((y_prop_list, prd.detach().cpu().numpy()), axis=0)
            y_prd_list = y_prd_list + torch.flatten(torch.argmax(prd, dim=-1)).cpu().numpy().tolist()
            y_true_list = y_true_list + torch.flatten(label).cpu().numpy().tolist()


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
        context.cell_type_prd_list = y_prd_list
        context.cell_type_true_list = y_true_list

        context.epoch_loss = np.mean(loss_sum)

    def generate_new_embedding(self, context: Optional[Context]):
        batch_size = context.batch_size
        pad_collate = context.pad_collate
        visual_save_path = context.visual_save_path
        data_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=pad_collate)
        self.model.eval()
        n_embedding, n_label = None, None
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
            print(f'attn score shape:{attn_score.shape}')

            if n_embedding is None:
                n_embedding = embedding.detach().cpu().squeeze(-2).numpy()
                n_label = label.cpu().numpy()

            else:
                n_embedding = np.concatenate((n_embedding, embedding.detach().cpu().squeeze(-2).numpy()), axis=0)
                n_label = np.concatenate((n_label, label.cpu().numpy()), axis=0)

        n_label = n_label.flatten()

        print(n_embedding.shape)
        print(n_label.shape)

    def show_attention_weights(self, context: Optional[Context]):
        batch_size = context.batch_size
        pad_collate = context.pad_collate
        # visual_save_path = context.visual_save_path
        data_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=pad_collate)
        self.model.eval()
        n_embedding = None
        n_attention_weights, n_label, n_feature_idx = None, None, None
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
                n_label = label.cpu().numpy()
                n_attention_weights = cell_attention_tmp
                # n_label = tissue_idx.cpu().numpy()
            else:
                n_embedding = np.concatenate((n_embedding, embedding.detach().cpu().squeeze(-2).numpy()), axis=0)
                n_label = np.concatenate((n_label, label.cpu().numpy()), axis=0)
                n_attention_weights = np.concatenate((n_attention_weights, cell_attention_tmp), axis=0)
                # n_label = np.concatenate((n_label, tissue_idx.cpu().numpy()), axis=0)
        print(f'n attention weights: {n_attention_weights.shape}')
        print(n_attention_weights.sum(-1))
        n_attention_weights_tensor = torch.tensor(n_attention_weights)
        vals, indices = n_attention_weights_tensor.topk(k=3, dim=-1, largest=True)
        vals = vals.numpy()
        print(vals)
        indices = indices.numpy()
        print(indices.shape)

        label_sets = set(n_label.flatten())
        n_label = n_label.flatten()
        cell_type_gene_dic = {}
        cell_vec = None
        word_idx_dic = context.word_idx_dic
        for l in label_sets:

            cell_vec_tmp = np.zeros(context.gene_num)
            cell_n_attention_weights_tensor = n_attention_weights_tensor[n_label == l, :].sum(0)
            vals, indices = cell_n_attention_weights_tensor.topk(k=context.k, dim=-1, largest=True)
            vals = vals.numpy()
            indices = indices.numpy()
            cell_gene_name_list = set(list(map(lambda x: word_idx_dic.getGene(x), indices)))
            cell_vec_tmp[indices] = vals
            cell_type_gene_dic[l] = cell_gene_name_list

            if cell_vec is None:
                cell_vec = cell_vec_tmp.reshape(1, -1)
            else:
                cell_vec = np.concatenate((cell_vec, cell_vec_tmp.reshape(1, -1)), axis=0)

        print(cell_type_gene_dic)
        write_file_to_pickle(cell_type_gene_dic, f'interpretable/{context.dataset_name}_top{context.k}_gene_dic.pk')


def show_embedding(dataset_filepath, d_model, head, d_ffw, dropout_rate, mapping_file, enhance_num, mlp_layer,
                   vocab, batch_size, lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,
                   save_model_path=None, trained_model_path=None, continue_train=False):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset_list, adata_list \
        = generate_dataset_list(filepath_list=dataset_filepath, word_idx_dic=word_idx_dic,
                                cell_type_idx_dic=cell_type_idx_dic)
    predict_type = len(cell_type_idx_dic.word2idx_dic)
    batch_set = 0
    for adata in adata_list:
        batch_set = set(adata.obs['batch_id'])
    print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    # predict_type = adata.uns['cell_type_nums']
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    print(f'batch set num: {len(batch_set)}')
    print(f'batch set: {batch_set}')
    model = generate_enhance_classification_model_with_d(d_model=d_model, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=len(cell_type_idx_dic.word2idx_dic), vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num, pred_num=len(batch_set))
    if trained_model_path is not None:
        state_dict = torch.load(trained_model_path)
        model.load_state_dict(state_dict['model'])
    trainer = ClassificationTrainer(model, train_dataset_list[0], [], continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)

    ctx = Context(batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None, epoch=None)
    ctx.visual_save_path = 'Tsne_class_PBMC45k_total_embedding_tissue18.jpg'
    trainer.generate_new_embedding(ctx)


def show_attention_weights(dataset_filepath, d_model, head, d_ffw, dropout_rate, mapping_file, enhance_num, mlp_layer,
                           vocab, batch_size, lr=0.001, device_name='cpu', random_seed=None, project_head_layer=None,
                           save_model_path=None, trained_model_path=None, continue_train=False, dataset_name='empty'):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    train_dataset_list, adata_list \
        = generate_dataset_list(filepath_list=dataset_filepath, word_idx_dic=word_idx_dic,
                                cell_type_idx_dic=cell_type_idx_dic)
    predict_type = len(cell_type_idx_dic.word2idx_dic)
    batch_set = [1]
    model = generate_enhance_classification_model_with_d(d_model=d_model, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=len(cell_type_idx_dic.word2idx_dic), vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num, pred_num=len(batch_set))
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
    ctx.k = 3
    trainer.show_attention_weights(ctx)




def train_enhance_class_model(train_filepath, test_size, epoch, d_model, head, d_ffw, dropout_rate,
                              mapping_file,
                              vocab, pca_num, batch_size, lr=0.001, device_name='cpu', random_seed=None,
                              mlp_layer=None,
                              save_model_path=None, trained_model_path=None, continue_train=False, freeze=False,
                              enhance_num=1):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset_list, test_dataset_list, adata_list \
        = generate_train_test_dataset_list(filepath_list=train_filepath, test_size=test_size,
                                           word_idx_dic=word_idx_dic, cell_type_idx_dic=cell_type_idx_dic,
                                           random_seed=random_seed)

    model = generate_enhance_classification_model_with_d(d_model=d_model,h_dim=d_model, head=head, d_ffw=d_ffw,
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

    trainer = ClassificationTrainer(model, total_train_dataset, test_dataset=None, continue_train=continue_train,
                                    trained_model_path=None, device_name=device_name, lr=lr)

    ctx = Context(epoch=epoch, batch_size=batch_size, save_model_path=save_model_path,
                  pad_collate=PadCollate(), random_seed=None)

    trainer.train(ctx)

    ctx = Context(pad_collate=PadCollate(), batch_size=batch_size)
    ctx.acc_list, ctx.ari_list, ctx.f1_scores_median_list, ctx.f1_scores_macro_list, ctx.f1_scores_micro_list, \
        ctx.f1_scores_weighted_list = [], [], [], [], [], []
    ctx.auc_list, ctx.best_auc, ctx.last_auc = [], 0, 0
    ctx.best_acc, ctx.best_ari, ctx.best_f1_scores_median, ctx.best_f1_scores_macro, ctx.best_f1_scores_micro, \
        ctx.best_f1_scores_weighted = 0, 0, 0, 0, 0, 0
    ctx.last_acc, ctx.last_ari, ctx.last_f1_scores_median, ctx.last_f1_scores_macro, ctx.last_f1_scores_micro, \
        ctx.last_f1_scores_weighted = 0, 0, 0, 0, 0, 0

    trainer.model.eval()
    with torch.no_grad():
        test_loader = DataLoader(dataset=total_test_dataset, batch_size=batch_size, shuffle=False,
                                                   collate_fn=PadCollate(), drop_last=False)
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
    return ctx


def train_enhance_class_model_with_d_from_pk(train_filepath, train_adata_path, label, test_size, epoch, d_model, head,
                                             d_ffw, dropout_rate,
                                             mapping_file,
                                             vocab, pca_num, batch_size, lr=0.001, device_name='cpu', random_seed=None,
                                             mlp_layer=None,
                                             save_model_path=None, trained_model_path=None, continue_train=False,
                                             freeze=False,
                                             project_layer=None, enhance_num=1, execute_model='train'):
    set_seed(random_seed)
    word_idx_dic, cell_type_idx_dic = read_mapping_file(mapping_file[0], mapping_file[1])
    train_dataset, test_dataset \
        = generate_train_test_dataset_list_from_pk(pk_filepath=train_filepath, label=label, test_size=test_size,
                                                   random_seed=random_seed)
    print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    # predict_type = adata.uns['cell_type_nums']
    print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    batch_set = [1]
    # print(f'word_idx_idc: {word_idx_dic.word2idx_dic}')
    # predict_type = adata.uns['cell_type_nums']
    # print(f'cell type dic: {cell_type_idx_dic.word2idx_dic}')
    print(f'batch set num: {len(batch_set)}')
    print(f'batch set: {batch_set}')
    # print(adata.obs['batch_id'])
    batch_set = [1]
    model = generate_enhance_classification_model_with_d(d_model=d_model, head=head, d_ffw=d_ffw,
                                                         dropout_rate=dropout_rate,
                                                         predict_type=len(cell_type_idx_dic.word2idx_dic), vocab=vocab,
                                                         device_name=device_name, mlp_layer=mlp_layer,
                                                         enhance_num=enhance_num, pred_num=len(batch_set))

    adata = check_anndata(train_adata_path)
    if trained_model_path is None:
        enhance_classification_construct(model, [adata], pca_num, word_idx_dic, device_name, vocab)
    else:
        model_state = torch.load(trained_model_path)
        model.encode.load_state_dict(model_state['model'])
    if freeze:
        model.encode.enhance_model_core.embedding.requires_grad_(False)
    gc.collect()

    #
    total_train_dataset = train_dataset
    total_test_dataset = test_dataset

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

def train_classification_model(test_size_list=None, dir_name='mouse', dataset_name='Bone_marrow',
                               word_dic_prefix='Bone_marrow', cell_type_prefix='Bone_marrow', times=1, print_postfix='',
                               print_prefix=None, enhance_num=1, head_num=1):
    if print_prefix is None:
        print_prefix = dataset_name
    if test_size_list is None:
        test_size_list = [0.9]
    auc_list = [[] for _ in range(len(test_size_list))]
    best_auc_list = [[] for _ in range(len(test_size_list))]
    last_auc_list = [[] for _ in range(len(test_size_list))]
    acc_list, ari_list, f1_scores_median_list, f1_scores_macro_list, f1_scores_micro_list, \
        f1_scores_weighted_list = [[] for _ in range(len(test_size_list))], [[] for _ in
                                                                             range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))]

    best_acc_list, best_ari_list, best_f1_scores_median_list, best_f1_scores_macro_list, best_f1_scores_micro_list, \
        best_f1_scores_weighted_list = [[] for _ in range(len(test_size_list))], [[] for _ in
                                                                                  range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))]
    last_acc_list, last_ari_list, last_f1_scores_median_list, last_f1_scores_macro_list, last_f1_scores_micro_list, \
        last_f1_scores_weighted_list = [[] for _ in range(len(test_size_list))], [[] for _ in
                                                                                  range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))]
    fine_tune_run_time_list, pretrain_run_time_list = [[] for _ in range(len(test_size_list))], \
        []
    for i in range(times):
        data_files = glob.glob(f'../../datasets/{dir_name}/{dataset_name}/*.h5ad')
        # data_files = glob.glob(f'../../../datasets/{dir_name}/mouse_{dataset_name}/*.h5ad')
        # data_files = glob.glob(f'../../../datasets/{dir_name}/{dataset_name}/*.h5ad')
        print(list(data_files))
        f = list(data_files)
        torch.cuda.synchronize()
        start = time.time()
        train_enhance_contrastive(dir_name=dir_name, dataset_name=dataset_name, word_dic_prefix=word_dic_prefix,
                                 cell_type_prefix=cell_type_prefix, enhance_num=enhance_num, head_num=head_num, random_seed=i)
        torch.cuda.synchronize()
        end = time.time()
        pretrain_run_time_list.append(end - start)
        for j in range(len(test_size_list)):
            start = time.time()
            torch.cuda.synchronize()
            ctx = train_enhance_class_model(train_filepath=f,
                                            test_size=test_size_list[j], epoch=40,
                                            freeze=True,
                                            mlp_layer=[],
                                            lr=0.001, enhance_num=enhance_num,
                                            mapping_file=[
                                                       f'../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                                                       f'../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
                                            save_model_path=f'pretrained/{dataset_name}_enhance{enhance_num}_{head_num}head_pretrained_class_model.pth',
                                            trained_model_path=f'pretrained/{dataset_name}_enhance{enhance_num}_{head_num}head_pretrained_cts_model.pth',
                                            d_model=64, head=head_num, d_ffw=64*3, dropout_rate=0.2, vocab=40000,
                                            pca_num=64,
                                            batch_size=100,
                                            device_name='cuda:0', random_seed=i, continue_train=False)
            torch.cuda.synchronize()
            end = time.time()
            fine_tune_run_time_list[j].append(end - start)
            acc_list[j].append(ctx.acc_list)
            auc_list[j].append(ctx.auc_list)
            ari_list[j].append(ctx.ari_list)
            f1_scores_median_list[j].append(ctx.f1_scores_median_list)
            f1_scores_macro_list[j].append(ctx.f1_scores_macro_list)
            f1_scores_micro_list[j].append(ctx.f1_scores_micro_list)
            f1_scores_weighted_list[j].append(ctx.f1_scores_weighted_list)

            best_acc_list[j].append(ctx.best_acc)
            best_auc_list[j].append(ctx.best_auc)
            best_ari_list[j].append(ctx.best_ari)
            best_f1_scores_median_list[j].append(ctx.best_f1_scores_median)
            best_f1_scores_macro_list[j].append(ctx.best_f1_scores_macro)
            best_f1_scores_micro_list[j].append(ctx.best_f1_scores_micro)
            best_f1_scores_weighted_list[j].append(ctx.best_f1_scores_weighted)

            last_acc_list[j].append(ctx.last_acc)
            last_auc_list[j].append(ctx.last_auc)
            last_ari_list[j].append(ctx.last_ari)
            last_f1_scores_median_list[j].append(ctx.last_f1_scores_median)
            last_f1_scores_macro_list[j].append(ctx.last_f1_scores_macro)
            last_f1_scores_micro_list[j].append(ctx.last_f1_scores_micro)
            last_f1_scores_weighted_list[j].append(ctx.last_f1_scores_weighted)
        print(list(data_files))
        gc.collect()
    print('END TRAIN!!!!!!!!!!!!!!')
    import sys
    savedStdout = sys.stdout  # 保存标准输出流
    print_log = open(f"log/{print_prefix}_printlog{print_postfix}.txt", "a")
    sys.stdout = print_log
    for j in range(len(test_size_list)):
        print(f'test size: {test_size_list[j]}')

        print(f'ctx acc list: {acc_list[j]}')
        print(f'ctx auc list: {auc_list[j]}')
        print(f'ctx ari list: {ari_list[j]}')
        print(f'ctx f1_scores_median list: {f1_scores_median_list[j]}')
        print(f'ctx f1_scores_macro list: {f1_scores_macro_list[j]}')
        print(f'ctx f1_scores_micro list: {f1_scores_micro_list[j]}')
        print(f'ctx f1_scores_weighted list: {f1_scores_weighted_list[j]}')

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


def train_enhance_with_d_from_pk(pk_filepath, adata_filepath,label, test_size_list=None, dir_name='mouse',
                                 dataset_name='Bone_marrow',
                                 word_dic_prefix='Bone_marrow', cell_type_prefix='Bone_marrow', times=1,
                                 print_postfix='',
                                 print_prefix=None, execute_model='train'):
    if print_prefix is None:
        print_prefix = dataset_name
    if test_size_list is None:
        test_size_list = [0.9]
    auc_list = [[] for _ in range(len(test_size_list))]
    best_auc_list = [[] for _ in range(len(test_size_list))]
    last_auc_list = [[] for _ in range(len(test_size_list))]
    acc_list, ari_list, f1_scores_median_list, f1_scores_macro_list, f1_scores_micro_list, \
        f1_scores_weighted_list = [[] for _ in range(len(test_size_list))], [[] for _ in
                                                                             range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))]

    best_acc_list, best_ari_list, best_f1_scores_median_list, best_f1_scores_macro_list, best_f1_scores_micro_list, \
        best_f1_scores_weighted_list = [[] for _ in range(len(test_size_list))], [[] for _ in
                                                                                  range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))]
    last_acc_list, last_ari_list, last_f1_scores_median_list, last_f1_scores_macro_list, last_f1_scores_micro_list, \
        last_f1_scores_weighted_list = [[] for _ in range(len(test_size_list))], [[] for _ in
                                                                                  range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))], \
        [[] for _ in range(len(test_size_list))], [[] for _ in range(len(test_size_list))]
    fine_tune_run_time_list, pretrain_run_time_list = [[] for _ in range(len(test_size_list))], \
        []
    torch.cuda.synchronize()
    start = time.time()
    train_enhance_contrastive_from_pk(pk_filepath=pk_filepath, adata_filepath=adata_filepath, enhance_num=1,
                                      dir_name=dir_name, word_dic_prefix=word_dic_prefix,
                                      cell_type_prefix=cell_type_prefix, dataset_name=dataset_name,
                                      head=1, random_seed=None)
    torch.cuda.synchronize()
    end = time.time()
    pretrain_run_time_list.append(end - start)
    for i in range(times):
        for j in range(len(test_size_list)):
            start = time.time()
            torch.cuda.synchronize()
            ctx = train_enhance_class_model_with_d_from_pk(train_filepath=pk_filepath,
                                                           train_adata_path=adata_filepath,
                                                           label=label,
                                                           # [1
                                                           #                         '../../datasets/mouse/Testis/mouse_Testis_total.h5ad',
                                                           #                   # ['../../datasets/mouse/Bladder/mouse_Bladder_total.h5ad']
                                                           #                             '../../datasets/mouse/Brain/mouse_Brain_total.h5ad',
                                                           #                             '../../datasets/mouse/Bladder/mouse_Bladder_total.h5ad'],
                                                           test_size=test_size_list[j], epoch=40,
                                                           freeze=True, mlp_layer=[128, 256],
                                                           project_layer=None,
                                                           lr=0.001, enhance_num=1,
                                                           mapping_file=[
                                                               f'../../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                                                               f'../../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
                                                           save_model_path=f'pretrained/{dataset_name}_tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
                                                           trained_model_path=f'pretrained/{dataset_name}_tissue_enhance1_1head_pretrained_cts_model_300_percent_pe_mean_with_tissue_without_D.pth',
                                                           d_model=64, head=1, d_ffw=192, dropout_rate=0.2, vocab=40000,
                                                           pca_num=64,
                                                           batch_size=32,
                                                           execute_model=execute_model,
                                                           device_name='cuda:0', random_seed=None, continue_train=False)
            torch.cuda.synchronize()
            end = time.time()
            fine_tune_run_time_list[j].append(end - start)
            acc_list[j].append(ctx.acc_list)
            auc_list[j].append(ctx.auc_list)
            ari_list[j].append(ctx.ari_list)
            f1_scores_median_list[j].append(ctx.f1_scores_median_list)
            f1_scores_macro_list[j].append(ctx.f1_scores_macro_list)
            f1_scores_micro_list[j].append(ctx.f1_scores_micro_list)
            f1_scores_weighted_list[j].append(ctx.f1_scores_weighted_list)

            best_acc_list[j].append(ctx.best_acc)
            best_auc_list[j].append(ctx.best_auc)
            best_ari_list[j].append(ctx.best_ari)
            best_f1_scores_median_list[j].append(ctx.best_f1_scores_median)
            best_f1_scores_macro_list[j].append(ctx.best_f1_scores_macro)
            best_f1_scores_micro_list[j].append(ctx.best_f1_scores_micro)
            best_f1_scores_weighted_list[j].append(ctx.best_f1_scores_weighted)

            last_acc_list[j].append(ctx.last_acc)
            last_auc_list[j].append(ctx.last_auc)
            last_ari_list[j].append(ctx.last_ari)
            last_f1_scores_median_list[j].append(ctx.last_f1_scores_median)
            last_f1_scores_macro_list[j].append(ctx.last_f1_scores_macro)
            last_f1_scores_micro_list[j].append(ctx.last_f1_scores_micro)
            last_f1_scores_weighted_list[j].append(ctx.last_f1_scores_weighted)
        gc.collect()
    print('END TRAIN!!!!!!!!!!!!!!')
    import sys
    savedStdout = sys.stdout  # 保存标准输出流
    print_log = open(f"result/{print_prefix}_printlog{print_postfix}.txt", "a")
    sys.stdout = print_log
    for j in range(len(test_size_list)):
        print(f'test size: {test_size_list[j]}')

        print(f'ctx acc list: {acc_list[j]}')
        print(f'ctx auc list: {auc_list[j]}')
        print(f'ctx ari list: {ari_list[j]}')
        print(f'ctx f1_scores_median list: {f1_scores_median_list[j]}')
        print(f'ctx f1_scores_macro list: {f1_scores_macro_list[j]}')
        print(f'ctx f1_scores_micro list: {f1_scores_micro_list[j]}')
        print(f'ctx f1_scores_weighted list: {f1_scores_weighted_list[j]}')

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

def show_enhance_embedding():
    dataset_name = 'Romanov'
    word_dic_prefix = 'Romanov'
    cell_type_prefix = 'Romanov'
    dir_name = 'cmp'
    adata_postfix = '.h5ad'
    data_files = glob.glob(f'../../../datasets/{dir_name}/{dataset_name}/*{adata_postfix}')
    print(list(data_files))
    f = list(data_files)
    show_embedding(
        trained_model_path='pretrained/Romanov_tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
        dataset_filepath=f, d_model=64, head=2, d_ffw=192, dropout_rate=0.2, enhance_num=1,
        mlp_layer=[100, 200],
        device_name="cuda:0",
        mapping_file=[f'../../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                      f'../../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
        vocab=40000, batch_size=100)


def show_enhance_attention_weights():
    dataset_name = 'Muscle'
    word_dic_prefix = 'Muscle'
    cell_type_prefix = 'Muscle'
    dir_name = 'mouse'
    adata_postfix = '.h5ad'
    data_files = glob.glob(f'../../../datasets/{dir_name}/{dataset_name}/*{adata_postfix}')
    print(list(data_files))
    f = list(data_files)
    show_attention_weights(
        trained_model_path=f'pretrained/{dataset_name}_tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
        dataset_filepath=f, d_model=64, head=1, d_ffw=192, dropout_rate=0.2, enhance_num=1,
        mlp_layer=[],
        device_name="cuda:0",
        mapping_file=[f'../../../datasets/preprocessed/{dir_name}/{word_dic_prefix}_word_dic.pk',
                      f'../../../datasets/preprocessed/{dir_name}/{cell_type_prefix}_cell_type_dic.pk'],
        vocab=40000, batch_size=100)



def train_PBMC160k():
    cell_annotation = pd.read_csv('../../../datasets/cmp/PBMC160k/GSE164378_sc.meta.data_3P.csv')
    cell_types = cell_annotation['celltype.l1'].values
    train_enhance_with_d_from_pk(pk_filepath='../../../datasets/PBMC160k/PBMC160k.pk',
                                 test_size_list=[0.9],
                                 print_postfix='_pretrain40epoch_finetune40epoch',
                                 adata_filepath='../../../datasets/cmp/PBMC160k/PBMC160kSample.h5ad',
                                 label=cell_types, dir_name='cmp', dataset_name='PBMC160k', word_dic_prefix='PBMC160k',
                                 cell_type_prefix='PBMC160k', times=1)

def train_scbloodnl():
    cell_annotation_v2 = pd.read_csv('../../../datasets/scbloodnl/1M_v2_cell_types.csv', sep='\t')
    cell_types_v2 = cell_annotation_v2[0].values.tolist()
    cell_annotation_v3 = pd.read_csv('../../../datasets/scbloodnl/1M_v3_cell_types.csv', sep='\t')
    cell_types_v3 = cell_annotation_v3[0].values.tolist()
    cell_types = cell_types_v2 + cell_types_v3
    train_enhance_with_d_from_pk(pk_filepath='../../../datasets/scbloodnl/scbloodnl.pk',
                                 test_size_list=[0.9],
                                 print_postfix='_pretrain40epoch_finetune40epoch',
                                 adata_filepath='../../../datasets/cmp/scbloodnl/scbloodnlSample.h5ad',
                                 label=cell_types, dir_name='cmp', dataset_name='scbloodnl', word_dic_prefix='scbloodnl',
                                 cell_type_prefix='scbloodnl', times=1)

