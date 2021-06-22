import torch
import torch.nn.functional as F

from tqdm import tqdm

from torch import nn
from torch_geometric.nn import GINConv



class Net(nn.Module):

    def __init__(self, num_features=1, hidden_dim=32, blocks=3, residual=False, jk=False):
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
        self.final_projection = nn.Linear(hidden_dim if not self.jk else blocks * hidden_dim, 1)

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


def initWeights(m, bias=0, lower_bound=-0.1, upper_bound=0.1):
    '''Auxiliary function that applies a uniform distribution to the weights and a bias=0.'''
    if type(m) == nn.Linear:
        m.weight.data.uniform_(lower_bound, upper_bound)
        m.bias.data.fill_(bias)


def train(model, optimizer, loader, device):
    '''
    Student train function.

    Parameters:
        - model: (nn.Module) Model to train on the given data.
        - optimizer: (torch.optim) Optimizer for training.
        - loader: (torch_geometric.data.dataloader.DataLoader) Torch data loader for training.
        - device: (torch.device) Destination device to perform the computations.
        
    Returns:
        - None
    '''
    # print()
    # print('-' * 30)
    # print('Init training...')
    model.train()
    loss = nn.MSELoss()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = loss(model(data), data.y)
        output.backward()
        optimizer.step()


def test(model, loader, device):
    '''
    Predict on unseen data using a given model.

    Parameters:
        - model: (nn.Module) Model to test on.
        - loader: (torch_geometric.data.dataloader.DataLoader) Torch data loader for testing.
        - device: (torch.device) Destination device to perform the computations.

    Returns:
        - (np.ndarray) Predictions of the model for all the test nodes.
    '''
    # print()
    # print('-' * 30)
    # print('Predicting data...')
    model.eval()
    predictions = []
    for data in loader:
        data = data.to(device)
        output = model(data)
        predictions.append(output.detach().numpy().reshape(-1))
    return predictions
