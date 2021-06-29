import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch import nn
from torch_geometric.nn import ChebConv



class Net(nn.Module):
    '''
    The chebyshev spectral graph convolutional operator from the “Convolutional Neural
    Networks on Graphs with Fast Localized Spectral Filtering” paper

                        X' = sum_{k=1}^K Z^{(k)} * Θ^{(k)}

    where Z^{(k)} is computed recursively by

                        Z^{(1)} = X
                        Z^{(2)} = L' * X
                        Z^{(k)} = 2 * L' * Z^{(k-1)} - Z^{(k-2)}

    and L' denotes the normalized Laplacian (2 * L / lambda_max) - I.
    '''
    def __init__(self, num_features=1, num_outputs=1, hidden_dim=32, blocks=3, K=2, normalization=None):
        super(Net, self).__init__()
        # Message passing blocks
        self.Cheb_blocks = nn.ModuleList([
            ChebConv(num_features if i == 0 else hidden_dim, hidden_dim, K=K, normalization=normalization) 
            for i in range(blocks)
        ])
        # Final projection layer for regression
        self.final_projection = nn.Linear(hidden_dim, num_outputs)

    def forward(self, data):
        '''
        You need to pass lambda_max to the forward method of
        this operator in case the normalization is non-symmetric.
        lambda_max should be a torch.Tensor of size
        [num_graphs] in a mini-batch scenario and a
        scalar/zero-dimensional tensor when operating on single graphs.
        You can pre-compute lambda_max via the
        torch_geometric.transforms.LaplacianLambdaMax transform.
        '''
        x, edge_index, lambda_max = data.x, data.edge_index, data.lambda_max
        # Forward through the Cheb blocks
        for i, cheb in enumerate(self.Cheb_blocks):
            x = F.relu(cheb(x, edge_index, lambda_max=lambda_max))
        return self.final_projection(x)
