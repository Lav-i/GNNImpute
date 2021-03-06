from .model import GNNImpute as Model
from .train import train
from .utils import adata2gdata, train_val_split, normalize


def GNNImpute(adata,
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
              verbose=True):
    input_dim = adata.n_vars

    model = Model(input_dim=input_dim, h_dim=512, z_dim=hidden, layerType=layer, heads=heads)

    adata = normalize(adata, filter_min_counts=False)
    adata = train_val_split(adata)
    gdata = adata2gdata(adata, use_raw=use_raw)

    train(gdata=gdata, model=model, no_cuda=no_cuda, epochs=epochs, lr=lr, weight_decay=weight_decay,
          patience=patience, fastmode=fastmode, verbose=verbose)

    pred = model(gdata['x'], gdata['adj'], gdata['size_factors'])

    adata.X = pred.detach().cpu()

    return adata
