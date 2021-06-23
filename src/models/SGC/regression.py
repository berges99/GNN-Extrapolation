import torch
import torch.nn.functional as F

from tqdm import tqdm

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
    def __init__(self, num_features=1, hidden_dim=32, K=3):
        super(Net, self).__init__()
        self.SGC_block = SGConv(num_features, 1, K=K)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        return self.SGC_block(x, edge_index)


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
