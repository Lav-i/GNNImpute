import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


def layer(layer_type, **kwargs):
    if layer_type == 'GCNConv':
        return GCNConv(in_channels=kwargs['in_channels'], out_channels=kwargs['out_channels'])
    elif layer_type == 'GATConv':
        return GATConv(in_channels=kwargs['in_channels'], out_channels=kwargs['out_channels'],
                       heads=kwargs['heads'], concat=kwargs['concat'], dropout=kwargs['dropout'])


class GNNImpute(torch.nn.Module):
    def __init__(self, input_dim, h_dim=512, z_dim=50, layerType='GATConv', heads=3):
        super(GNNImpute, self).__init__()

        #### Encoder ####
        self.encode_conv1 = layer(layerType, in_channels=input_dim, out_channels=h_dim,
                                  heads=heads, concat=False, dropout=0.6)
        self.encode_bn1 = torch.nn.BatchNorm1d(h_dim)

        self.encode_conv2 = layer(layerType, in_channels=h_dim, out_channels=z_dim,
                                  heads=heads, concat=False, dropout=0.6)
        self.encode_bn2 = torch.nn.BatchNorm1d(z_dim)

        #### Decoder ####
        self.decode_linear1 = torch.nn.Linear(z_dim, h_dim)
        self.decode_bn1 = torch.nn.BatchNorm1d(h_dim)

        self.decode_linear2 = torch.nn.Linear(h_dim, input_dim)

    def encode(self, x, edge_index):
        x = F.relu(self.encode_bn1(self.encode_conv1(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.encode_bn2(self.encode_conv2(x, edge_index)))
        x = F.dropout(x, p=0.5, training=self.training)

        return x

    def decode(self, x):
        x = F.relu(self.decode_bn1(self.decode_linear1(x)))
        x = F.relu(self.decode_linear2(x))

        return x

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x = self.decode(z)
        return x
