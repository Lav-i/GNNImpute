# GNNImpute

## Installation

### Option 1: Use python virtual environment with conda

Download the source code from github.
```shell script
git clone git@github.com:Lav-i/GNNImpute.git
cd GNNImpute
```

Create a python virtual environment and install the required packages. If your device is cuda available, you can choose to use torch with gpu.
```shell script
conda create -n gnnimpute python=3.6
conda activate gnnimpute
pip install -r requirements.txt
```

### Option 2: Use docker

TODO

## Prepare data

### Download

* [MOUSE Embryo Stem Cells Klein](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525)
    * File: GSE65525_RAW.tar
    * Matrix size: (2717, 24175)
    * Matrix size(after preprocess)：(2713, 24021)
    * Clusters：4

```shell script
wget https://www.ncbi.nlm.nih.gov/geo/download/\?acc\=GSE65525\&format\=file -O ./data/Klein/GSE65525_RAW.tar
tar xvf ./data/Klein/GSE65525_RAW.tar -C ./data/Klein
```

* [Human Frozen PBMCs (Donor A) 10X](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/frozen_pbmc_donor_a)
    * File：Gene / cell matrix (filtered)
    * Matrix size: (2900, 32738)
    * Matrix size(after preprocess)：(2843, 13003)
    * Clusters: ?
    
```shell script
wget https://cf.10xgenomics.com/samples/cell-exp/1.1.0/frozen_pbmc_donor_a/frozen_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz -O ./data/PBMC/frozen_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
tar xvf ./data/PBMC/frozen_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz -C ./data/PBMC
mv ./data/PBMC/filtered_matrices_mex/hg19/* ./data/PBMC
```

### Preprocess

Process Klein data set into standard format.
```shell script
python ./data/Klein/preprocess.py
```

Process PBMC data set into standard format.
```shell script
python ./data/PBMC/preprocess.py
```

Output file is the filtered expression matrix, the file format is h5ad.

### Mask (Get benchmark data)

Mask Klein data set.
```shell script
python ./data/mask.py --masked_prob=0.1 --dataset=Klein
```

Mask PBMC data set.
```shell script
python ./data/mask.py --masked_prob=0.1 --dataset=PBMC
```

Output folder contains the main output file (representing the masked expression matrix) in h5ad and csv formats. And the file in npz format indicates the location of the dropout event.

## Train

Impute Klein data set.
```shell script
python ./train.py --layer=GATConv --masked_prob=0.1 --heads=3 --dataset=Klein
```

Impute PBMC data set.
```shell script
python ./train.py --layer=GATConv --masked_prob=0.1 --heads=3 --dataset=PBMC
```

Output folder contains the main output file (representing the imputed expression matrix) in h5ad and csv formats.

