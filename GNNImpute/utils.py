import torch
import numpy as np
import networkx as nx

from torch_geometric.data import Data
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


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


def kneighbor(adata, n_components=50, k=5):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(adata.X.A)

    A = kneighbors_graph(data_pca, k, mode='connectivity', include_self=False)
    G = nx.from_numpy_matrix(A.todense())

    edges = []
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])

    edges = np.array(edges).T

    return edges


def adata2gdata(adata, use_raw=True):
    edges = kneighbor(adata, n_components=50, k=5)

    edges = torch.tensor(edges, dtype=torch.long)
    features = torch.tensor(adata.X.A, dtype=torch.float)
    labels = torch.tensor(adata.X.A, dtype=torch.float)
    if use_raw:
        labels = torch.tensor(adata.raw.X.A, dtype=torch.float)

    gdata = Data(x=features, y=labels, edge_index=edges)
    gdata.train_mask = torch.tensor(adata.obs.idx_train, dtype=torch.bool)
    gdata.val_mask = torch.tensor(adata.obs.idx_val, dtype=torch.bool)

    return gdata
