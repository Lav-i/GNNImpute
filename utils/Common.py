import copy
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


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


if __name__ == '__main__':
    pass
