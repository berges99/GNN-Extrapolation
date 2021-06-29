import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GINConv



class Net(nn.Module):
    '''
    The graph isomorphism operator from the “How Powerful are Graph Neural Networks?” paper

                        X' = h_Θ((A + (1 + ε) * I) * X)

    where h_Θ denotes a neural network, .i.e. an MLP.
    '''
    def __init__(self, num_features=1, num_outputs=1, hidden_dim=32, blocks=3, residual=False, jk=False):
        super(Net, self).__init__()
        # Additional model configurations parameters
        self.residual = residual
        self.jk = jk
        # Message passing blocks
        self.GIN_blocks = nn.ModuleList([
            GINConv(
                nn.Sequential(
                    nn.Linear(num_features if i == 0 else hidden_dim, hidden_dim), 
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, hidden_dim)
            ))
            for i in range(blocks)
        ])
        # Final projection layer for regression
        self.final_projection = nn.Linear(hidden_dim if not self.jk else blocks * hidden_dim, num_outputs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.residual: residual = x
        if self.jk: cat = []
        # Forward through the GIN blocks
        for i, gin in enumerate(self.GIN_blocks):
            x = F.relu(gin(x, edge_index))
            if self.residual:
                x += residual
                residual = x
            if self.jk: cat.append(x)
        if self.jk: x = torch.cat(cat, dim=1)
        return self.final_projection(x)
