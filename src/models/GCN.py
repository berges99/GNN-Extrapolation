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
    def __init__(self, 
                 num_features=1,
                 num_outputs=1, 
                 hidden_dim=32, 
                 blocks=3, 
                 residual=0, 
                 jk=False, 
                 pre_linear=False):
        super(Net, self).__init__()
        # Additional model configurations parameters
        self.residual = residual
        self.jk = jk
        self.pre_linear = pre_linear
        # Pre-linear layer that projects data to hidden_dim
        if self.pre_linear:
            self.lin0 = nn.Linear(num_features, hidden_dim)
        # Message passing blocks
        self.GCN_blocks = nn.ModuleList([
            GCNConv(num_features if i == 0 and not self.pre_linear else hidden_dim, hidden_dim) 
            for i in range(blocks)
        ])
        # Final projection layer/s for regression
        self.lin1 = nn.Linear(blocks * hidden_dim if self.jk else hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_outputs)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.jk: cat = []
        # Forward through pre_linear layer if specified
        if self.pre_linear:
            x = F.relu(self.lin0(x))
        # Forward through the GIN blocks
        for i, gcn in enumerate(self.GCN_blocks):
            out = F.relu(gcn(x, edge_index))
            # If no initial residual connection is to be added
            if i == 0 and self.residual != 2:
                x = out
            else:
                x = out + x if self.residual in [1, 2] else out
            if self.jk: cat.append(x)
        if self.jk: x = torch.cat(cat, dim=1)
        # Forward throught the final projections (MLP with 2 layers)
        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
