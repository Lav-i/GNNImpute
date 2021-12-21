from torch_geometric.nn import GCNConv, GATConv


def layer(layer_type, **kwargs):
    if layer_type == 'GCNConv':
        return GCNConv(in_channels=kwargs['in_channels'], out_channels=kwargs['out_channels'])
    elif layer_type == 'GATConv':
        return GATConv(in_channels=kwargs['in_channels'], out_channels=kwargs['out_channels'],
                       heads=kwargs['heads'], concat=kwargs['concat'], dropout=kwargs['dropout'])
