# %%
import os
import json
import time
import glob
import pandas as pd
import scanpy as sc
import networkx as nx
from scipy import sparse

import torch
from torch_geometric.data import Data

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from collections import namedtuple

from model import GNNImpute
from utils.Common import *

# torch.cuda.set_device(0)

# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--layer', default='GATConv', type=str)
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--masked_prob', default=0.1, type=float)
parser.add_argument('--epochs', default=3000, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--hidden', default=50, type=int)
parser.add_argument('--patience', default=200, type=int)
parser.add_argument('--fastmode', default=False, type=bool)
parser.add_argument('--heads', default=3, type=int)
parser.add_argument('--downsample', default=1, type=float)
parser.add_argument('--split', default=[0.6, 0.2, 0.2], nargs='+', type=float)
parser.add_argument('--dataset', default='Klein', type=str)

args = parser.parse_args()

# Args = namedtuple('args', ['layer', 'no_cuda', 'masked_prob', 'epochs',
#                            'lr', 'weight_decay', 'hidden', 'patience',
#                            'fastmode', 'heads', 'downsample', 'split', 'dataset'])
#
# args = Args(
#     # layer='GCNConv',
#     layer='GATConv',
#     no_cuda=False,
#     masked_prob=0.1,
#     epochs=3000,
#     lr=0.001,
#     weight_decay=0.0005,
#     hidden=50,
#     patience=200,
#     fastmode=False,
#     heads=3,
#     downsample=1.,
#     split=[0.6, 0.2, 0.2],
#     dataset='PBMC'
#     # dataset='Campbell'
#     # dataset='Chen'
#     # dataset='Klein'
#     # dataset='Zeisel'
# )

assert args.layer in ['GCNConv', 'GATConv']
assert args.dataset in ['PBMC', 'Klein']

device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

adata = sc.read_h5ad(
    './data/%s/masked/%s_%s.h5ad' % (args.dataset, args.dataset, str(args.masked_prob).replace('.', '')))

maskIndex = sparse.load_npz(
    './data/%s/masked/%s_maskIndex_%s.csv.npz' % (args.dataset, args.dataset, str(args.masked_prob).replace('.', '')))

split = args.split
test_val = np.random.choice(adata.shape[0], int(adata.shape[0] * (split[1] + split[2])), replace=False)
idx_train = [i for i in list(range(adata.shape[0])) if i not in test_val]
idx_test = np.random.choice(test_val, int(len(test_val) * (split[2] / (split[1] + split[2]))), replace=False)
idx_val = [i for i in test_val if i not in idx_test]

masking_row_train, masking_col_train = np.where(maskIndex.A[idx_train, :] > 0)
masking_row_val, masking_col_val = np.where(maskIndex.A[idx_val, :] > 0)
masking_row_test, masking_col_test = np.where(maskIndex.A[idx_test, :] > 0)

tmp = np.zeros(adata.shape[0], dtype=bool)
tmp[idx_train] = True
idx_train = tmp
tmp = np.zeros(adata.shape[0], dtype=bool)
tmp[idx_val] = True
idx_val = tmp
tmp = np.zeros(adata.shape[0], dtype=bool)
tmp[idx_test] = True
idx_test = tmp

# %%

pca = PCA(n_components=50)
data_pca = pca.fit_transform(adata.X.A)

k = 5
A = kneighbors_graph(data_pca, k, mode='connectivity', include_self=False)
G = nx.from_numpy_matrix(A.todense())

edges = []
for (u, v) in G.edges():
    edges.append([u, v])
    edges.append([v, u])

edges = np.array(edges).T

# %%

edges = torch.tensor(edges, dtype=torch.long)
features_masked = torch.tensor(adata.X.A, dtype=torch.float)
labels = torch.tensor(adata.raw.X.A, dtype=torch.float)

data = Data(x=features_masked, y=labels, edge_index=edges)
data.train_mask = torch.tensor(idx_train, dtype=torch.bool)
data.val_mask = torch.tensor(idx_val, dtype=torch.bool)
data.test_mask = torch.tensor(idx_test, dtype=torch.bool)

# %%


model = GNNImpute(input_dim=data.num_features, z_dim=args.hidden, layerType=args.layer, heads=args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lossFunc = torch.nn.MSELoss(reduction='mean')
data = data.to(device)


# %%


def train_wrapper(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    pred = model(data.x, data.edge_index)

    dropout_pred = pred[data.train_mask]
    dropout_true = data.y[data.train_mask]

    loss_train = lossFunc(dropout_pred, dropout_true)

    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        pred = model(data.x, data.edge_index)

    dropout_pred = pred[data.val_mask]
    dropout_true = data.y[data.val_mask]

    loss_val = lossFunc(dropout_pred, dropout_true)

    if (epoch + 1) % 10 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


# %%


def test_wrapper():
    model.eval()

    pred = model(data.x, data.edge_index)

    dropout_pred = pred[data.test_mask]
    dropout_true = data.y[data.test_mask]

    loss = lossFunc(dropout_pred, dropout_true)

    print('Test set results:', 'loss= {:.4f}'.format(loss.item()))

    y = dropout_true[masking_row_test, masking_col_test].detach().cpu().numpy()
    h = dropout_pred[masking_row_test, masking_col_test].detach().cpu().numpy()

    mse = mean_squared_error(y, h)
    mae = mean_absolute_error(y, h)
    pearsonr = pearsonr_error(y, h)
    cosine_similarity = cosine_similarity_score(y, h)

    clus = {}
    if 'cluster' in adata.obs:
        clusters = adata.obs.cluster.values

        adata_pred = sc.AnnData(pred.detach().cpu().numpy())

        kmeans = KMeans(n_clusters=len(set(clusters))).fit(adata_pred.X)
        ari = float('%.4f' % adjusted_rand_score(clusters, kmeans.labels_))
        nmi = float('%.4f' % normalized_mutual_info_score(clusters, kmeans.labels_))

        clus = {
            'ari': ari,
            'nmi': nmi
        }

    return {
               **clus,
               'mse': float('%.4f' % mse),
               'mae': float('%.4f' % mae),
               'pearsonr': float('%.4f' % pearsonr),
               'cosine_similarity': float('%.4f' % cosine_similarity)
           }, pred.detach().cpu().numpy()


# %%

t_total = time.time()
loss_values = []
bad_counter = 0
best = float('inf')
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train_wrapper(epoch))

    if loss_values[-1] < best:
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print('Total time elapsed: {:.4f}s'.format(time.time() - t_total))

# Restore best model
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
result, imputedData = test_wrapper()

path = './data/%s/imputed' % args.dataset

if not os.path.exists(path):
    os.makedirs(path)

pd.DataFrame(imputedData, index=adata.obs.index, columns=adata.var.index) \
    .T.to_csv(path + '/%s_%s.csv' % (args.dataset, str(args.masked_prob).replace('.', '')))

adata.X = sparse.csr_matrix(imputedData)
adata.write(path + '/%s_%s.h5ad' % (args.dataset, str(args.masked_prob).replace('.', '')))

print(json.dumps({
    'layer': args.layer,
    'masked_prob': args.masked_prob,
    'heads': args.heads,
    'downsample': args.downsample,
    'dataset': args.dataset,
    'best_epoch': best_epoch,
    'split': args.split,
    **result
}))
