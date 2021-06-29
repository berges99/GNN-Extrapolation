import torch

from torch import nn
from torch_geometric.nn import SGConv



class Net(nn.Module):
    '''
    The simple graph convolutional operator from the “Simplifying Graph Convolutional
    Networks” paper

                        X' = (D'^(-1/2) * A' * D'^(-1/2))^K * X * Θ

    where A' denotes the adjacency matrix with inserted self-loops and D' its diagonal 
    degree matrix.

    The SGC reduces the entire procedure to a simple feature propagation step.
    '''
    def __init__(self, num_features=1, num_outputs=1, hidden_dim=32, K=3):
        super(Net, self).__init__()
        self.SGC_block = SGConv(num_features, num_outputs, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.SGC_block(x, edge_index)
