import gc
import glob
from typing import Optional

import numpy as np
import torch.optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, ConcatDataset
import tqdm

from models.model import ClassificationModel, LabelSmoothing
import torch.nn as nn

from models.utils import save_model, loss_visual, set_seed


class Context():
    def __init__(self, epoch=0, batch_size=1, save_model_path=None, pad_collate=None, random_seed=None,
                 execute_model='train'):
        self.epoch = epoch
        self.batch_size = batch_size
        self.save_model_path = save_model_path
        self.pad_collate = pad_collate
        self.random_seed = random_seed
        self.execute_model = execute_model


class Trainer():
    def __init__(self, model: nn.Module, train_dataset, test_dataset, device_name='cpu',
                 lr=0.001, weight_decay=1e-2, trained_model_path=None, continue_train=False):
        self.model = model
        self.device = torch.device(device_name)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.drop_last = False
        if trained_model_path is not None:
            state_dict = torch.load(trained_model_path)
            self.model.load_state_dict(state_dict['model'])
            print("using trained model")
            if continue_train:
                self.opt.load_state_dict(state_dict['opt'])
                print("continue training")

    def replace_dataset(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self, context: Optional[Context]):
        train_loader_list, test_loader_list = [], []
        self.save_model_path = context.save_model_path
        batch_size = context.batch_size
        pad_collate = context.pad_collate
        epoch = context.epoch
        if type(self.train_dataset) is not list:
            train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                      pin_memory=True, collate_fn=pad_collate, drop_last=False)
            train_loader_list.append(train_loader)
            print(f'train datasets size: {len(self.train_dataset)}')
        else:
            for i in range(len(self.train_dataset)):
                train_loader_list.append(DataLoader(dataset=self.train_dataset[i], batch_size=batch_size, shuffle=True,
                                                    collate_fn=pad_collate, drop_last=False))
                print(f'train datasets {i + 1} size: {len(self.train_dataset[i])}')

        if type(self.test_dataset) is not list:
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                     pin_memory=True, collate_fn=pad_collate, drop_last=False)
            test_loader_list.append(test_loader)
            print(f'test datasets size: {len(self.test_dataset)}')
        else:
            for i in range(len(self.test_dataset)):
                test_loader_list.append(DataLoader(dataset=self.test_dataset[i], batch_size=batch_size, shuffle=True,
                                                   collate_fn=pad_collate, drop_last=False))
                print(f'test datasets {i + 1} size: {len(self.test_dataset[i])}')

        train_info_list = [[] for _ in range(len(train_loader_list))]
        test_info_list = [[] for _ in range(len(test_loader_list))]
        for c_epoch in tqdm.tqdm(range(epoch)):
            context.c_epoch = c_epoch
            self.model.train()
            for i in range(len(train_loader_list)):
                self.train_inner(train_loader_list[i], context)
                train_info_list[i].append(context.epoch_loss)
            # if context.execute_model is not None and context.execute_model == 'skip_test':
            #     continue
            self.model.eval()
            for i in range(len(test_loader_list)):
                with torch.no_grad():
                    self.test_inner(test_loader_list[i], context)
                    test_info_list[i].append(context.epoch_loss)
            # gc.collect()
        context.train_info_list = train_info_list
        context.test_info_list = test_info_list
        self.on_train_test_completed(context)

    def train_inner(self, train_loader, context: Optional[Context]):
        pass

    def test_inner(self, test_loader, context: Optional[Context]):
        pass

    def on_train_test_completed(self, context: Optional[Context]):
        self.info_solve(context)
        if self.save_model_path is not None:
            save_model(model=self.model, opt=self.opt, model_path=self.save_model_path)
        pass

    def info_solve(self, context: Optional[Context]):
        train_info_list, test_info_list = context.train_info_list, context.test_info_list
        loss_total, test_loss_total = [], []
        print(train_info_list)
        print(test_info_list)
        for i in range(len(train_info_list)):
            for j in range(len(train_info_list[i])):
                loss_total.append(train_info_list[i][j])

        for i in range(len(test_info_list)):
            for j in range(len(test_info_list[i])):
                test_loss_total.append(test_info_list[i][j])
        loss_visual(loss_total, test_loss_total)
        loss_total.clear()
        test_loss_total.clear()
        pass

def enhance_classification_construct(model: ClassificationModel, adata_list, pca_num, word_idx_idc, device_name, vocab):
    pca = PCA(n_components=pca_num)
    new_embedding = np.zeros((vocab, pca_num))
    for adata in adata_list:
        gene_vec = pca.fit_transform(adata.X.T)
        print(gene_vec.shape)
        gene_names = adata.var_names
        gene_idx = gene_names.map(lambda x: word_idx_idc.getIdx(x.lower()) if word_idx_idc.getIdx(x.lower()) else -1)
        gene_idx = np.array(gene_idx).astype(int)
        print(f'gene idx: {gene_idx}')
        print(f'gene idx: {gene_idx[gene_idx >= 0]}')
        new_embedding[gene_idx[gene_idx >= 0]] = gene_vec[gene_idx >= 0]
    model.encode.embedding.feature_embedding.word_embed = \
        torch.nn.Embedding.from_pretrained(torch.tensor(new_embedding, dtype=torch.float32,
                                                        device=torch.device(device_name)), freeze=False, padding_idx=0)
    print('using pca embedding')
    print(model.encode.embedding.feature_embedding.word_embed.weight.data)


if __name__ == '__main__':
    print(torch.__version__)
    # test()

    # train_class_model(filepath=['../../datasets/mouse/Bladder/mouse_Bladder_total.h5ad',
    #                             '../../datasets/mouse/Brain/mouse_Brain_total.h5ad',
    #                             '../../datasets/mouse/Te stis/mouse_Testis_total.h5ad'], test_size=0.2, epoch=300,
    #                   d_model=50, head=5, d_ffw=100, dropout_rate=0.2, vocab=30000, pca_num=50, batch_size=512,
    #                   device_name='cuda:0', random_seed=123)

    # {'Stromal cell': 0, 'Vascular endothelial cell': 1, 'Basal epithelial cell': 2, 'Endothelial cell': 3, 'Dendritic cell': 4, 'Natural killer cell': 5, 'Urothelium': 6, 'Smooth muscle cell': 7, 'Umbrella cell': 8, 'Mesenchymal stromal cell': 9, 'Vascular smooth muscle progenitor cell': 10, 'Epithelial cell': 11, 'Macrophage': 12, 'Basophil': 13, 'Erythrocyte progenitor cell': 14, 'Mast cell': 15, 'Hematopoietic stem progenitor cell': 16, 'Megakaryocyte progenitor cell': 17, 'Multipotent progenitor cell': 18, 'B cell': 19, 'Neutrophil progenitor cell': 20, 'Pre-pro B cell': 21, 'Monocyte progenitor cell': 22, 'Monocyte': 23, 'Eosinophil progenitor cell': 24, 'Erythroblast': 25, 'T cell': 26, 'Neutrophil': 27, 'Granulocyte': 28, 'Pan-GABAergic': 29, 'Astrocyte': 30, 'Astroglial cell': 31, 'Microglia': 32, 'Oligodendrocyte precursor cell': 33, 'Myelinating oligodendrocyte': 34, 'Hypothalamic ependymal cell': 35, 'Neuron': 36, 'Schwann cell': 37, 'Skeletal muscle cell': 38, 'Reproductive tissues': 39, 'Neuronal progenitor cell': 40, 'Muscle cell': 41, 'Ganglion cell': 42, 'Lymphoid progenitor cell': 43, 'Erythroid cell': 44, 'Progenitor cell': 45, 'Granule neuron': 46, 'Postmitotic neuron': 47, 'Hippocampus neuron': 48, 'Pyramidal neuron': 49, 'Ependymal cell': 50, 'Purkinje cell': 51, 'Radial glia': 52, 'Erythrocyte': 53, 'Dopaminergic neuron': 54, 'Neural progenitor cell': 55, 'Metanephric mesenchyme': 56, 'Intercalated cell': 57, 'Fenestrated endothelial cell': 58, 'Proximal tubule cell': 59, 'Adipocyte': 60, 'Proximal tubule brush border cell': 61, 'Glomerular epithelial cell': 62, 'Thick ascending limb of the loop of Henle': 63, 'S1 proximal tubule cell': 64, 'Distal collecting duct principal cell': 65, 'Ureteric epithelium': 66, 'Distal convoluted tubule': 67, 'Periportal hepatocyte': 68, 'Hepatocyte': 69, 'Pericentral hepatocyte': 70, 'Kuppfer cell': 71, 'AT1 cell': 72, 'Ciliated cell': 73, 'Nuocyte': 74, 'Clara cell': 75, 'Dividing T cell': 76, 'AT2 cell': 77, 'Dividing cell': 78, 'Ig<U+2212>producing B cell': 79, 'Alveolar bipotent progenitor cell': 80, 'Eosinophil granulocyte': 81, 'Plasmacytoid dendritic cell': 82, 'Conventional dendritic cell': 83, 'Dividing dendritic cell': 84, 'Neutrophil granulocyte': 85, 'Interstitial macrophage': 86, 'Alveolar macrophage': 87, 'Luminal progenitor cell': 88, 'Myoepithelial cell': 89, 'Ductal luminal cell': 90, 'Secretory alveoli cell': 91, 'Myeloid leukocyte': 92, 'Stem and progenitor cell': 93, 'Luminal cell': 94, 'Muscle progenitor cell': 95, 'Granulocyte monocyte progenitor cell': 96, 'Dividng cell': 97, 'Osteoblast': 98, 'Preosteoblast/Osteoblast/Bone cell/Cartilage cell': 99, 'Left ventricle cardiomyocyte': 100, 'Ventricle cardiomyocyte': 101, 'Cardiac muscle cell': 102, 'Atrial cardiomyocyte': 103, 'Tendon stem/progenitor cell': 104, 'Glial cell': 105, 'Chondrocyte': 106, 'Brown adipose tissue': 107, 'Mesenchymal cell': 108, 'Acinar cell': 109, 'Lymphatic vessel endothelial cell': 110, 'Natural killer T cell': 111, 'Ductal cell': 112, 'Cartilage cell': 113, 'Oligodendrocyte': 114, 'Osteoclast': 115, 'Melanocyte': 116, 'Keratinocyte': 117, 'Ovarian vascular surface endothelium cell': 118, 'Small luteal cell': 119, 'Granulosa cell': 120, 'Ovarian surface epithelium cell': 121, 'luteal cell': 122, 'Thecal cell': 123, 'Large luteal cell': 124, 'Cumulus cell': 125, 'Endocrine cell': 126, 'Beta cell': 127, 'Hematopoietic stem cell and progenitor cell': 128, 'Trophoblast progenitor cell': 129, 'Labyrinthine trophoblast': 130, 'Invasive spongiotrophoblast': 131, 'Progenitor trophoblast': 132, 'Endodermal cell': 133, 'Spiral artery trophoblast giant cell': 134, 'PE lineage cell': 135, 'Spongiotrophoblast': 136, 'Decidual stromal cell': 137, 'Glandular epithelium': 138, 'Prostate gland cell': 139, 'Epithelium of small intestinal villi': 140, 'Columnar epithelium': 141, 'S cell': 142, 'Paneth cell': 143, 'Marginal zone B cell': 144, 'Plasma cell': 145, 'G cell': 146, 'Antral mucous cell': 147, 'Parietal cell': 148, 'Stomach cell': 149, 'Tuft cell': 150, 'Pit cell': 151, 'Gastric mucosal cell': 152, 'Leydig cell': 153, 'Preleptotene spermatogonia': 154, 'Elongating spermatid': 155, 'Pre-Sertoli cell': 156, 'Spermatid': 157, 'Spermatogonia': 158, 'Sertoli cell': 159, 'Spermatocyte': 160, 'gdT cell': 161, 'abT cell': 162, 'Pre T cell': 163, 'Proliferating thymocyte': 164, 'DPT cell': 165}
