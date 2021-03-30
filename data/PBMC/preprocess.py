# %%
import os
import numpy as np
import scanpy as sc

# %%

adata = sc.read_10x_mtx('./data/PBMC/', var_names='gene_symbols', cache=True)

# %%

adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.n_genes_by_counts < 2000, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.raw = adata

# %%

folder = os.path.exists('./data/PBMC/processed')

if not folder:
    os.makedirs('./data/PBMC/processed')

adata.write('./data/PBMC/processed/PBMC.h5ad')

# %%

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata)

marker_genes = ['S100A9', 'GZMH', 'HLA-DRB5', 'RP11-290F20.3', 'CD7', 'LTB', 'LYZ', 'RPS5', 'CD74', 'GZMA', 'RPS8',
                'FCER1G', 'RPL32', 'GNLY', 'S100A8', 'B2M', 'LST1', 'RPS13', 'HLA-DQA1', 'RPL11', 'S100A10', 'RPLP2',
                'RPS2', 'S100A6', 'S100A4', 'LYAR', 'HLA-DRB1', 'AIF1', 'CCL5', 'TYROBP', 'CD52', 'IL7R', 'CTSW',
                'HLA-DPB1', 'CLIC3', 'CD79B', 'FTH1', 'HLA-DPA1', 'CST3', 'RPL31', 'FTL', 'RPL13', 'FXYD5', 'RPS6',
                'CD79A', 'GZMK', 'NKG7', 'HLA-B', 'IL32', 'HLA-DRA']

sc.pl.heatmap(adata, marker_genes, groupby='leiden', dendrogram=True, swap_axes=True, use_raw=True)
