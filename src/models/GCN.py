import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GCNConv



class Net(nn.Module):
    '''
    The graph convolutional operator from the “Semi-supervised Classification with Graph 
    Convolutional Networks” paper

                        X' = D'^(-1/2) * A' * D'^(-1/2) * X * Θ

    where A' denotes the adjacency matrix with inserted self-loops and D' its diagonal 
    degree matrix.
    '''
    def __init__(self, num_features=1, num_outputs=1, hidden_dim=32, blocks=3, residual=False, jk=False, mlp=1):
        super(Net, self).__init__()
        # Additional model configurations parameters
        self.residual = residual
        self.jk = jk
        # Message passing blocks
        self.GCN_blocks = nn.ModuleList([
            GCNConv(num_features if i == 0 else hidden_dim, hidden_dim) 
            for i in range(blocks)
        ])
        # Final projection layer/s for regression
        self.final_projections = nn.ModuleList([
            nn.Linear(
                blocks * hidden_dim if self.jk and i == 0 else hidden_dim, 
                num_outputs if i == mlp - 1 else hidden_dim)
            for i in range(mlp)
        ])


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.residual: residual = x
        if self.jk: cat = []
        # Forward through the GIN blocks
        for i, gcn in enumerate(self.GCN_blocks):
            x = F.relu(gcn(x, edge_index))
            if self.residual:
                x += residual
                residual = x
            if self.jk: cat.append(x)
        if self.jk: x = torch.cat(cat, dim=1)
        for linear_i in self.final_projections:
            x = linear_i(x)
        return x
