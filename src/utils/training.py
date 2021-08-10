import torch
import numpy as np

from torch import nn



##################
# INIT FUNCTIONS #
##################


def initWeightsUniform(m, bias=0.0, lower_bound=-0.1, upper_bound=0.1):
    '''Auxiliary function that applies a uniform distribution to the weights and a bias=0.'''
    if type(m) == nn.Linear:
        m.weight.data.uniform_(lower_bound, upper_bound)
        m.bias.data.fill_(bias)


def initWeightsXavier(m, bias=0.0, gain=1.0):
    '''Auxiliary funtion that applies a xavier uniform distribution to the weights and a bias=0.'''
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(bias)


def initWeightsDefault(m):
    '''Abstract function for system robustness with default initializations (does nothing).'''
    pass


##############
# REGRESSION #
##############


def train_regression(model, optimizer, scheduler, loader, device):
    '''
    Student train function.

    Parameters:
        - model: (nn.Module) Model to train on the given data.
        - optimizer: (torch.optim) Optimizer for training.
        - scheduler
        - loader: (torch_geometric.data.dataloader.DataLoader) Torch data loader for training.
        - device: (torch.device) Destination device to perform the computations.
        
    Returns:
        - None
    '''
    # print()
    # print('-' * 30)
    # print('Init training...')
    model.train()
    loss = nn.MSELoss() # reduction='mean' -> the loss is the average SE per graph
    losses = []
    for data in loader:
        # Handle pre-transforms of the SIGN network
        if 'x1' in data:
            K = model.K
            xs = [data.x.to(device)]
            xs += [data[f'x{i}'].to(device) for i in range(1, K + 1)]
            optimizer.zero_grad()
            output = loss(model(xs), data.y)
        else:
            data = data.to(device)
            optimizer.zero_grad()
            output = loss(model(data), data.y)
        losses.append(float(output.detach().numpy()))
        output.backward()
        optimizer.step()
    # Onche the epoch is completed, perform the update of the scheduler
    # scheduler.step()
    # Reduce on plateau scheduler assumed
    scheduler.step(np.mean(np.array(losses)))


def test_regression(model, loader, device):
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
        # Handle pre-transforms of the SIGN network
        if 'x1' in data:
            K = model.K
            xs = [data.x.to(device)]
            xs += [data[f'x{i}'].to(device) for i in range(1, K + 1)]
            output = model(xs)
        else:
            data = data.to(device)
            output = model(data)
        predictions.append(output.detach().numpy().reshape(-1))
    return predictions


##################
# CLASSIFICATION #
##################


# TBD
def train_classification(model, optimizer, scheduler, loader, device):
    '''
    Student train function.

    Parameters:
        - model: (nn.Module) Model to train on the given data.
        - optimizer: (torch.optim) Optimizer for training.
        - scheduler
        - loader: (torch_geometric.data.dataloader.DataLoader) Torch data loader for training.
        - device: (torch.device) Destination device to perform the computations.
        
    Returns:
        - None
    '''
    # print()
    # print('-' * 30)
    # print('Init training...')
    return None


# TBD 
def test_classification(model, loader, device):
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
    return None
