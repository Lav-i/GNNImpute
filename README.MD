## GNNImpute

### Prepare data

#### Download

* [MOUSE Embryo Stem Cells Klein](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525)
    * File: GSE65525_RAW.tar
    * Matrix size: (2717, 24175)
    * Matrix size(after preprocess)：(2713, 24021)
    * Clusters：4

* [Human Frozen PBMCs (Donor A) 10X](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/frozen_pbmc_donor_a)
    * File：Gene / cell matrix (filtered)
    * Matrix size: (2900, 32738)
    * Matrix size(after preprocess)：(2843, 13003)
    * Clusters: ?

#### Preprocess
    
```shell script
python ./data/Klein/preprocess.py
```

Output file is the filtered expression matrix, the file format is h5ad.

#### Mask

```shell script
python ./data/mask.py --masked_prob=0.1 --dataset=Klein
```

Output folder contains the main output file (representing the masked expression matrix) in h5ad and csv formats. And the file in npz format indicates the location of the dropout event.

### Train

```shell script
python ./train.py --layer=GATConv --masked_prob=0.1 --heads=3 --dataset=Klein
```


