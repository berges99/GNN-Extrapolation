import torch
import torch.nn.functional as F

from tqdm import tqdm

from torch import nn



class Net(nn.Module):
    '''
    The Scalable Inception Graph Neural Network module (SIGN) from the "SIGN: Scalable Inception Graph 
    Neural Networks" paper, which precomputes the fixed representations

                        X^(i) = (D'^(-1/2) * A' * D'^(-1/2))^i * X

    for i in {1, ..., K} and saves them in data.x1, data.x2, ...
    '''
    def __init__(self, num_features=1, hidden_dim=32, K=3):
        super(Net, self).__init__()
        # Linear layers
        self.K = K
        self.lins = nn.ModuleList([
            nn.Linear(num_features, hidden_dim) for _ in range(K + 1)
        ])
        # Final projection layer for regression
        self.final_projection = nn.Linear((K + 1) * hidden_dim, 1)

    def forward(self, xs):
        hs = []
        for x, lin in zip(xs, self.lins):
            h = F.relu(lin(x))
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        return self.final_projection(h)


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
    K = model.K
    model.train()
    loss = nn.MSELoss()
    for data in loader:
        xs = [data.x.to(device)]
        xs += [data[f'x{i}'].to(device) for i in range(1, K + 1)]
        optimizer.zero_grad()
        output = loss(model(xs), data.y)
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
    K = model.K
    model.eval()
    predictions = []
    for data in loader:
        xs = [data.x.to(device)]
        xs += [data[f'x{i}'].to(device) for i in range(1, K + 1)]
        output = model(xs)
        predictions.append(output.detach().numpy().reshape(-1))
    return predictions
