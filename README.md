# scTrans
This is a single cell transformer-based cell type annotation method.

## Requirements
+ Python >= 3.7
+ torch >= 1.13.1

## scRNA seq-data preprocess
All single cell RNA sequence data should be log transformed and keep genes expressed are expressed at least in one cell.
adata is a scanpy anndata object, its var_names must be gene symbols.
```py
import scanpy as sc
from models.utils import check_anndata
adata = check_anndata("single_cell_h5ad_file_path")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.filter_genes(adata, min_cells=1)
```

## pretrain process
Unsupervised pre-training using unlabeled single-cell datasets
The pretrain phase is as follows
```py
from scTrans_core import scTrans_core
from models.utils import check_anndata
pretrain_adata = check_anndata("pretrain_file_path")
#create scTrans core
save_path = 'file_save_path'
sc_core = scTrans_core(file_save_path=save_path, device_name='cuda')
sc_core.generate_mapping_file_for_embedding(gene_name_list=pretrain_adata.var_names.tolist())
config = {
    'embedding_dim':64, #gene ebmedding dim
    'd_model':64,
    'h_dim':64,
    'head':1,           #attention head
    'd_ffw':64 * 3,
    'dropout_rate':0.2,
    'vocab':60000,      #vocab size, must larger than gene number
    'embedding_dropout':False,
    'enhance_num':1,    #encoder block number
    }
sc_core.create_encoder(config)
#embedding initializing from pretrain adata based on pca, perform pca on all cells when sample is None.
#if sample is Integer, Select some samples for PCA to obtain gene embedding, which can be used when running on large-scale data
sc_core.embedding_initialize_from_pca([pretrain_adata], sample=None)
sc_core.pre_train([pretrain_adata], epoch=40)
```
## finetune process
Fine-tune the pre-trained model using labeled datasets.
```py 
from scTrans_core import scTrans_core
from models.utils import check_anndata
save_path = 'file_save_path'
fine_tune_adata = check_anndata("pretrained_model_file_path")
cell_type_list = fine_tune_adata.obs['cell_type'].unique().tolist()
cell_type_number = len(cell_type_list)

sc_core = scTrans_core(file_save_path=save_path, device_name='cuda')
encoder_config = {
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
sc_core.create_encoder(encoder_config)
cls_config = {
    'd_model': 64,
    'predict_type': cell_type_number,
    'dropout_rate': 0.2
}
sc_core.create_classification_model(cls_config)
sc_core.load_pretrained_model(saved_model_path=f'{save_path}/finetuned_model.pth', type='pretrained')
sc_core.generate_cell_type_mapping_file(cell_type_list=cell_type_list)
sc_core.fine_tune([fine_tune_adata], epoch=40)
```

## predict process
Use the fine-tuned model to predict the cell type of the query dataset
```py
from scTrans_core import scTrans_core
from models.utils import check_anndata
import pandas as pd
save_path = 'file_save_path'
query_adata = check_anndata('query_file_path')
sc_core = scTrans_core(file_save_path=save_path, device_name='cuda')
encoder_config = {
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
sc_core.create_encoder(encoder_config)
cls_config = {
    'd_model': 64,
    'predict_type': 'cell_type_number',
    'dropout_rate': 0.2
}
sc_core.create_classification_model(cls_config)

sc_core.load_pretrained_model(saved_model_path=f'{save_path}/finetuned_model.pth', type='finetuned')

sc_core.predict_cell_type(query_adata=query_adata, batch_size=100)
predict_results_df = pd.DataFrame({'cell_type': query_adata.obs['cell_type'].tolist(), 'cell_type_predict': query_adata.obs['cell_type_predict'].tolist()})
predict_results_df.to_csv(f'{save_path}/results.csv', index=True)
```

## generate cell embedding
Extract low-dimensional embeddings from the query dataset using the fine-tuned model
```py
from scTrans_core import scTrans_core
from models.utils import check_anndata
import scanpy as sc
save_path = 'file_save_path'
query_adata = check_anndata('query_file_path')
sc_core = scTrans_core(file_save_path=save_path, device_name='cuda')
encoder_config = {
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
sc_core.create_encoder(encoder_config)
cls_config = {
    'd_model': 64,
    'predict_type': 'cell_type_number',
    'dropout_rate': 0.2
}
sc_core.create_classification_model(cls_config)

sc_core.load_pretrained_model(saved_model_path=f'{save_path}/finetuned_model.pth', type='finetuned')
sc_core.generate_embedding(query_adata=query_adata)
sc_core.predict_cell_type(query_adata=query_adata, batch_size=100)
embedding = query_adata.obsm['embedding']
print(embedding.shape)
sc.pp.neighbors(query_adata, use_rep='embedding')
sc.tl.umap(query_adata)
sc.pl.umap(query_adata, color='cell_type_predict', show=False,
           save=f"{save_path}/cell_type_predict.png")
```

## generate attention weights
Generate the weight of each gene in each cell and store it as anndata data
```py 
from scTrans_core import scTrans_core
from models.utils import check_anndata
import scanpy as sc
save_path = 'file_save_path'
query_adata = check_anndata('query_file_path')
sc_core = scTrans_core(file_save_path=save_path, device_name='cuda')
encoder_config = {
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
sc_core.create_encoder(encoder_config)
cls_config = {
    'd_model': 64,
    'predict_type': 'cell_type_number',
    'dropout_rate': 0.2
}
sc_core.create_classification_model(cls_config)

sc_core.load_pretrained_model(saved_model_path=f'{save_path}/finetuned_model.pth', type='finetuned')
query_adata_attention_weights_adata = sc_core.generate_attetion_weights(query_adata)
```