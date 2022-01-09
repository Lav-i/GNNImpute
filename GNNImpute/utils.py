import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    # if size_factors or normalize_input or logtrans_input:
    #     adata.raw = adata.copy()
    # else:
    #     adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def train_val_split(adata, train_size=0.6, val_size=0.2, test_size=0.2):
    assert train_size + val_size + test_size == 1

    adata = adata.copy()

    cell_nums = adata.n_obs
    test_val = np.random.choice(cell_nums, int(cell_nums * (val_size + test_size)), replace=False)
    idx_train = [i for i in list(range(cell_nums)) if i not in test_val]
    idx_test = np.random.choice(test_val, int(len(test_val) * (test_size / (val_size + test_size))), replace=False)
    idx_val = [i for i in test_val if i not in idx_test]

    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_train] = True
    adata.obs['idx_train'] = tmp
    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_val] = True
    adata.obs['idx_val'] = tmp
    tmp = np.zeros(cell_nums, dtype=bool)
    tmp[idx_test] = True
    adata.obs['idx_test'] = tmp

    return adata


def row_normalize(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def kneighbor(adata, n_components=50, k=5):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(adata.X)

    A = kneighbors_graph(data_pca, k, mode='connectivity', include_self=True)

    return row_normalize(A)


def adata2gdata(adata, use_raw=True):
    adj = kneighbor(adata, n_components=50, k=5)

    adj = torch.tensor(adj.A, dtype=torch.float)
    features = torch.tensor(adata.X, dtype=torch.float)
    labels = torch.tensor(adata.X, dtype=torch.float)
    size_factors = torch.tensor(adata.obs.size_factors, dtype=torch.float).reshape(-1, 1)
    if use_raw:
        labels = torch.tensor(adata.raw.X.A, dtype=torch.float)

    train_mask = torch.tensor(adata.obs.idx_train, dtype=torch.bool)
    val_mask = torch.tensor(adata.obs.idx_val, dtype=torch.bool)

    return {
        'x': features,
        'y': labels,
        'size_factors': size_factors,
        'adj': adj,
        'train_mask': train_mask,
        'val_mask': val_mask
    }
