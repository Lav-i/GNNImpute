# %%
import os
import copy
import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse


# %%

# def mask(data_train, masked_prob):
#     """
#     将表达矩阵中非零的值随机置为0并返回，同时返回置为0的元素的坐标
#     :param data_train: 表达矩阵
#     :param masked_prob: 置0比例
#     :return:
#     """
#     index_pair_train = np.where(data_train != 0)
#     masking_idx_train = np.random.choice(index_pair_train[0].shape[0], int(index_pair_train[0].shape[0] * masked_prob),
#                                          replace=False)
#     # to retrieve the position of the masked: data_train[index_pair_train[0][masking_idx], index_pair[1][masking_idx]]
#     X_train = copy.deepcopy(data_train)
#     X_train[index_pair_train[0][masking_idx_train], index_pair_train[1][masking_idx_train]] = 0
#     return X_train, index_pair_train[0][masking_idx_train], index_pair_train[1][masking_idx_train]


def maskPerCol(data_train, masked_prob):
    """
    将表达矩阵中每列非零的值随机置为0并返回，同时返回置为0的元素的坐标
    :param data_train: 表达矩阵
    :param masked_prob: 置0比例
    :return:
    """
    X_train = copy.deepcopy(data_train)
    rows = []
    cols = []
    for col in range(data_train.shape[1]):
        index_pair_train = np.where(data_train[:, col])
        if index_pair_train[0].shape[0] <= 3:
            continue
        masking_idx_train = np.random.choice(index_pair_train[0].shape[0],
                                             int(index_pair_train[0].shape[0] * masked_prob),
                                             replace=False)
        X_train[index_pair_train[0][masking_idx_train], [col] * masking_idx_train.shape[0]] = 0
        for i in index_pair_train[0][masking_idx_train]:
            rows.append(i)
            cols.append(col)

    return X_train, rows, cols


# %%

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--masked_prob', default=0.1, type=float)
parser.add_argument('--dataset', default='Klein', type=str)
parser.add_argument('--downsample', default=1.0, type=float)

args = parser.parse_args()

adata = sc.read_h5ad('./data/%s/processed/%s.h5ad' % (args.dataset, args.dataset))
sc.pp.normalize_total(adata)
adata.raw = adata

# %%

path = './data/%s/masked' % args.dataset

if not os.path.exists(path):
    os.makedirs(path)

masked, masking_row, masking_col = maskPerCol(adata.raw.X.A, args.masked_prob)

pd.DataFrame(masked, index=adata.obs.index, columns=adata.var.index) \
    .T.to_csv(path + '/%s_%s.csv' % (args.dataset, str(args.masked_prob).replace('.', '')))

adata.X = sparse.csr_matrix(masked)
adata.write(path + '/%s_%s.h5ad' % (args.dataset, str(args.masked_prob).replace('.', '')))

# %%

maskIndex = sparse.coo_matrix(([1] * len(masking_col), (masking_row, masking_col)))

sparse.save_npz(path + '/%s_maskIndex_%s.csv' % (args.dataset, str(args.masked_prob).replace('.', '')), maskIndex)
