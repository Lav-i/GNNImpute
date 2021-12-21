# %%
import numpy as np
import scanpy as sc
from scipy import sparse

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from GNNImpute.api import GNNImpute

# %%

adata = sc.read_h5ad('../data/Klein/masked/Klein_01.h5ad')

maskIndex = sparse.load_npz('../data/Klein/masked/Klein_maskIndex_01.csv.npz')


def pearsonr_error(y, h):
    res = []
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
        h = h.reshape((1, -1))

    for i in range(y.shape[0]):
        res.append(pearsonr(y[i], h[i])[0])
    return np.mean(res)


def cosine_similarity_score(y, h):
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
        h = h.reshape((1, -1))
    cos = cosine_similarity(y, h)
    res = []
    for i in range(len(cos)):
        res.append(cos[i][i])
    return np.mean(res)


# %%

adata = GNNImpute(adata=adata,
                  layer='GATConv',
                  no_cuda=False,
                  epochs=3000,
                  lr=0.001,
                  weight_decay=0.0005,
                  hidden=50,
                  patience=200,
                  fastmode=False,
                  heads=3,
                  use_raw=True,
                  verbose=True)

# %%

dropout_pred = adata.X[adata.obs.idx_test]
dropout_true = adata.raw.X.A[adata.obs.idx_test]

masking_row_test, masking_col_test = np.where(maskIndex.A[adata.obs.idx_test, :] > 0)

y = dropout_true[masking_row_test, masking_col_test]
h = dropout_pred[masking_row_test, masking_col_test]

mse = float('%.4f' % mean_squared_error(y, h))
mae = float('%.4f' % mean_absolute_error(y, h))
pcc = float('%.4f' % pearsonr_error(y, h))
cs = float('%.4f' % cosine_similarity_score(y, h))

# %%

clusters = adata.obs.cluster.values

adata_pred = sc.AnnData(adata.X)

sc.pp.normalize_total(adata_pred)
sc.pp.log1p(adata_pred)
sc.pp.highly_variable_genes(adata_pred, n_top_genes=2000)
adata_pred = adata_pred[:, adata_pred.var.highly_variable]
sc.pp.scale(adata_pred, max_value=10)

kmeans = KMeans(n_clusters=len(set(clusters))).fit(adata_pred.X)
ari = float('%.4f' % adjusted_rand_score(clusters, kmeans.labels_))
nmi = float('%.4f' % normalized_mutual_info_score(clusters, kmeans.labels_))

# %%

print(mse, mae, pcc, cs, ari, nmi)
