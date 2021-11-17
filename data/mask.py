# %%
import os
import sys
import pandas as pd
import scanpy as sc

from scipy import sparse
from warnings import simplefilter

sys.path.append('./')
from utils.Common import maskPerCol

simplefilter(action='ignore', category=FutureWarning)

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
