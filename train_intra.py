import os.path

from datasets.preprocess import generate_mapping_file
from models.impl.ClassificationTrainer_intra import train_enhance_class_model
from models.impl.ContrastiveTrainer import train_enhance_contrastive_model
from models.utils import check_anndata


def preprocess(adata_path, save_dic_path):
    filename, _ = os.path.splitext(os.path.basename(adata_path))
    generate_mapping_file([adata_path], f'{save_dic_path}/{filename}_word_dic.pk',
                          f'{save_dic_path}/{filename}_cell_type_dic.pk')


def pre_train(adata_path, block_num=1, head_num=1):
    filename, _ = os.path.splitext(os.path.basename(adata_path))
    train_enhance_contrastive_model(train_filepath=[adata_path],
                                    with_d=False,
                                    freeze=False,
                                    epoch=40,
                                    lr=0.0002,
                                    enhance_num=block_num,
                                    project_head_layer=None,
                                    mapping_file=[
                                        f'preprocessed/{filename}_word_dic.pk',
                                        f'preprocessed/{filename}_cell_type_dic.pk'],
                                    save_model_path=f'pretrained/{filename}_enhance{block_num}_{head_num}head_pretrained_cts_model.pth',
                                    d_model=64, h_dim=64, head=head_num, d_ffw=64 * 3, dropout_rate=0.2, vocab=40000,
                                    pca_num=64,
                                    batch_size=100,
                                    device_name='cuda:0',
                                    continue_train=False)


def fine_tune(adata_path, test_size=0.9, block_num=1, head_num=1):
    filename, _ = os.path.splitext(os.path.basename(adata_path))
    ctx = train_enhance_class_model(train_filepath=[adata_path],
                                    test_size=test_size, epoch=40,
                                    freeze=True,
                                    mlp_layer=[],
                                    lr=0.001, enhance_num=block_num,
                                    mapping_file=[
                                               f'preprocessed/{filename}_word_dic.pk',
                                               f'preprocessed/{filename}_cell_type_dic.pk'],
                                    save_model_path=f'pretrained/{filename}_enhance{block_num}_{head_num}head_pretrained_class_model.pth',
                                    trained_model_path=f'pretrained/{filename}_enhance{block_num}_{head_num}head_pretrained_cts_model.pth',
                                    d_model=64, head=head_num, d_ffw=64 * 3, dropout_rate=0.2, vocab=40000,
                                    pca_num=64,
                                    batch_size=100,
                                    device_name='cuda:0',
                                    continue_train=False)
    accuracy, f1_macro = ctx.last_acc, ctx.last_f1_scores_macro
    predict_result = ctx.cell_type_prd_list
    true_result = ctx.cell_type_true_list
    print(f'accuracy:{accuracy}, f1-macro:{f1_macro}')
    print(f'predict result:{predict_result}')
    print(f'true result:{true_result}')


if __name__ == '__main__':
    preprocess('datasets/mouse/Muscle/Muscle.h5ad', 'preprocessed')
    pre_train(adata_path='datasets/mouse/Muscle/Muscle.h5ad')
    fine_tune(adata_path='datasets/mouse/Muscle/Muscle.h5ad', test_size=0.9)
