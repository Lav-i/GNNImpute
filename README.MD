
dowmload Klein(https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525) 

tar xvf GSE65525_RAW.tar

python ./data/Klein/preprocess.py

python ./data/mask.py --masked_prob=0.1 --dataset=Klein

python ./train.py --layer=GATConv --masked_prob=0.1 --heads=3 --dataset=Klein

