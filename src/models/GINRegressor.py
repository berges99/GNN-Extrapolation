import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GINConv #, global_add_pool


class GIN(nn.Module):

    def __init__(self, num_features, hidden_dim):
        super(GIN, self).__init__()

        nn1 = nn.Sequential(nn.Linear(num_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        #self.bn2 = nn.BatchNorm1d(hidden_dim)

        nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.conv3 = GINConv(nn3)
        #self.bn3 = nn.BatchNorm1d(num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        #residual = x

        x = self.conv1(x, edge_index)
        #x += residual
        #residual = x
        #x = self.bn1(x)
        x = self.conv2(x, edge_index)
        #x += residual
        #x = self.bn2(x)
        x = self.conv3(x, edge_index)
        #x += residual
        #x = self.bn3(x)
        return x