# %%
import os
import scanpy as sc
from scipy import sparse

# %%

adataD0 = sc.read_csv('./data/Klein/GSM1599494_ES_d0_main.csv.bz2')
adataD2 = sc.read_csv('./data/Klein/GSM1599497_ES_d2_LIFminus.csv.bz2')
adataD4 = sc.read_csv('./data/Klein/GSM1599498_ES_d4_LIFminus.csv.bz2')
adataD7 = sc.read_csv('./data/Klein/GSM1599499_ES_d7_LIFminus.csv.bz2')

# %%

adata = sc.AnnData.concatenate(adataD0.T, adataD2.T, adataD4.T, adataD7.T, batch_key='cluster',
                               batch_categories=['d0', 'd2', 'd4', 'd7', ])
adata.X = sparse.csr_matrix(adata.X)

# %%

sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

adata = adata[adata.obs.total_counts < 75000, :]

# sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
# sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=False, multi_panel=True)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.raw = adata

# %%

folder = os.path.exists('./data/Klein/processed')

if not folder:
    os.makedirs('./data/Klein/processed')

adata.write('./data/Klein/processed/Klein.h5ad')
