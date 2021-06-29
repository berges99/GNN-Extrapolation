import torch
import torch.nn.functional as F

from torch import nn



class Net(nn.Module):
    '''
    The Scalable Inception Graph Neural Network module (SIGN) from the "SIGN: Scalable Inception Graph 
    Neural Networks" paper, which precomputes the fixed representations

                        X^(i) = (D'^(-1/2) * A' * D'^(-1/2))^i * X

    for i in {1, ..., K} and saves them in data.x1, data.x2, ...
    '''
    def __init__(self, num_features=1, num_outputs=1, hidden_dim=32, K=3, pre_linear=False):
        super(Net, self).__init__()
        self.K = K
        # Use pre-individual linear layers for each of the channels (x1, x2, ...)
        self.pre_linear = pre_linear
        if pre_linear:
            self.lins = nn.ModuleList([
                nn.Linear(num_features, hidden_dim) for _ in range(K + 1)
            ])
        # Final projection layer for regression
        self.final_projection = nn.Linear(
            (K + 1) * num_features if not pre_linear else (K + 1) * hidden_dim, num_outputs)

    def forward(self, xs):
        hs = []
        if self.pre_linear:
            for x, lin in zip(xs, self.lins):
                h = F.relu(lin(x))
                hs.append(h)
        else:
            hs = xs
        h = torch.cat(hs, dim=-1)
        return self.final_projection(h)
