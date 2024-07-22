# scTrans
This is a single cell transformer-based cell type annotation method.

## Requirements
+ Python >= 3.7
+ torch >= 1.13.1

## scRNA seq-data preprocess
All single cell RNA sequence data should be log transformed and keep genes expressed are expressed at least in one cell.
```py
import scanpy as sc
sc.pp.log1p(adata)
sc.pp.filter_genes(adata, min_cells=1)
```

## Cross batch annotation
- [Cross Batch annotation](train_crossbatch_multi_reference.py)
- We give a cross batch annotation example in this file, including preprocessing, pre training fine-tuning, and generating cell embeddings.

## Intra annotation
- [Intra annotation](train_intra.py)
- This example was the annotation task of 31 tissue in mouse cell atlas.
- All data are used for pre-training, 10% stratified sampling cells are used for fine-tuning, and the remaining 90% of cells for testing.
