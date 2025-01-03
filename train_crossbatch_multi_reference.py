import gc
import glob
import os

from models.impl.ClassificationTrainer_crossbatch import train_enhance_class_model_with_extra, show_embedding
from models.impl.ContrastiveTrainer import train_enhance_contrastive_model
from models.utils import generate_mapping_file


def preprocess(reference_adata_file_list, query_data_name, save_path='preprocessed'):
    generate_mapping_file(reference_adata_file_list, f'{save_path}/{query_data_name}_reference_word_dic.pk',
                          f'{save_path}/{query_data_name}_reference_cell_type_dic.pk')


def pre_train(train_adata_file_list, word_dic_prefix, cell_type_prefix, save_model_prefix, block_num=1, head_num=1):
    train_enhance_contrastive_model(train_filepath=train_adata_file_list,
                                    with_d=False,
                                    epoch=40,
                                    lr=0.0002,
                                    enhance_num=block_num,
                                    project_head_layer=None,
                                    embedding_dropout=True,
                                    mapping_file=[
                                        f'preprocessed/{word_dic_prefix}_word_dic.pk',
                                        f'preprocessed/{cell_type_prefix}_cell_type_dic.pk'],
                                    save_model_path=f'pretrained/{save_model_prefix}_tissue_enhance{block_num}_{head_num}head_pretrained_cts_model.pth',
                                    d_model=64, h_dim=64, head=head_num, d_ffw=64 * 3, dropout_rate=0.2, vocab=40000,
                                    pca_num=64,
                                    batch_size=100,
                                    device_name='cuda:0',
                                    random_seed=0,
                                    continue_train=False)


def fine_tune(train_adata_file_path_list, test_file_path_list, word_dic_prefix, cell_type_prefix, save_model_prefix,
              block_num=1, head_num=1):
    ctx = train_enhance_class_model_with_extra(train_filepath=train_adata_file_path_list,
                                         test_filepath=test_file_path_list, epoch=40,
                                         freeze=False,
                                         embedding_dropout=True,
                                         mlp_layer=[],
                                         lr=0.001, enhance_num=block_num,
                                         mapping_file=[
                                             f'preprocessed/{word_dic_prefix}_word_dic.pk',
                                             f'preprocessed/{cell_type_prefix}_cell_type_dic.pk'],
                                         save_model_path=f'pretrained/{save_model_prefix}_tissue_enhance{block_num}_{head_num}head_pretrained_class_model.pth',
                                         trained_model_path=f'pretrained/{save_model_prefix}_tissue_enhance{block_num}_{head_num}head_pretrained_cts_model.pth',
                                         d_model=64, h_dim=64, head=head_num, d_ffw=64 * 3, dropout_rate=0.2,
                                         vocab=40000,
                                         pca_num=64,
                                         batch_size=100,
                                         device_name='cuda:0',
                                         continue_train=False,
                                         random_seed=None)
    accuracy, f1_macro = ctx.last_acc, ctx.last_f1_scores_macro
    predict_result = ctx.cell_type_prd_list
    true_result = ctx.cell_type_true_list
    print(f'accuracy:{accuracy}, f1-macro:{f1_macro}')
    return accuracy, f1_macro

    # print(f'predict result:{predict_result}')
    # print(f'true result:{true_result}')

def generate_embedding(dataset_file_path_list, trained_model_prefix, word_dic_prefix, cell_type_prefix,
                       block_num=1, head_num=1, anndata_postfix=''):
    show_embedding(
        # trained_model_path=f'pretrained/{model_dataset_name}_tissue_enhance1_1head_pretrained_class_model_300_percent_pe_mean_with_tissue_without_D.pth',
        trained_model_path=f'pretrained/{trained_model_prefix}_tissue_enhance{block_num}_{head_num}head_pretrained_cts_model.pth',
        dataset_filepath=dataset_file_path_list,
        d_model=64, head=1, d_ffw=64*3, dropout_rate=0.2, enhance_num=1,
        mlp_layer=[],
        title_name='',
        anndata_postfix=anndata_postfix,
        device_name="cuda:0",
        mapping_file=[
            f'preprocessed/{word_dic_prefix}_word_dic.pk',
            f'preprocessed/{cell_type_prefix}_cell_type_dic.pk'],
        vocab=40000, batch_size=100)



if __name__ == '__main__':
    # need generate h5ad first
    reference_data_list = ['datasets/Mouse-Pancreas-Baron/Baron.h5ad', 'datasets/Mouse-Pancreas-MCA/MCA-Pancreas.h5ad']
    query_data_list = ['datasets/Mouse-Pancreas-TMS/TMS-Pancreas.h5ad']
    query_data_name = 'TMS-Pancreas'
    preprocess(reference_data_list, query_data_name)
    pre_train(train_adata_file_list=reference_data_list, word_dic_prefix=f'{query_data_name}_reference',
              cell_type_prefix=f'{query_data_name}_reference', save_model_prefix=f'{query_data_name}_reference')
    accuracy, f1_macro = fine_tune(train_adata_file_path_list=reference_data_list, test_file_path_list=query_data_list,
              word_dic_prefix=f'{query_data_name}_reference',
              cell_type_prefix=f'{query_data_name}_reference',
              save_model_prefix=f'{query_data_name}_reference')

    generate_embedding(dataset_file_path_list=query_data_list, trained_model_prefix=f'{query_data_name}_reference',
                       word_dic_prefix=f'{query_data_name}_reference',
                       cell_type_prefix=f'{query_data_name}_reference')



