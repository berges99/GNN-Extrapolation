import time
import torch
import argparse
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch import nn
from torch_geometric.data import DataLoader

from utils.io import *
from utils.convert import *
from models.GINRegressor import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--filename', '-f', type=str, default='', help='Filename of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/preferential_attachment', help='Default path to the data.')
    return parser.parse_args()


def initWeights(m):
	'''Auxiliary function that applies a uniform distribution to the weights and a bias=0.'''
	if type(m) == nn.Linear:
		m.weight.data.uniform_(-0.3, 0.3)
		m.bias.data.fill_(0)


def test(model, loader):
	'''
	Predict on unseen data.

	Parameters:
		- model: (models.GINRegressor.GIN)
		- loader: (torch_geometric.data.dataloader.DataLoader) Torch data loader for testing.

	Returns:
		- (np.ndarray) Predictions of the model for all the test nodes.
	'''
	print()
	print('-' * 30)
	print('Generating outputs for all nodes in the dataset...')
	model.eval()
	predictions = []
	for data in tqdm(loader):
		data = data.to(device)
		output = model(data)
		# pred = output.max(dim=1)[1]
		predictions.append(output.detach().numpy().reshape(-1))
	return predictions


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(f'{args.path}/raw')
	# Read the dataset and convert it to torch_geometric.data
	networkx_dataset = readPickle(f'{args.path}/raw/{filename}')
	torch_dataset = fromNetworkx(networkx_dataset, add_degree=True)
	torch_dataset_loader = DataLoader(torch_dataset, batch_size=1)
	# Init the model
	model = GIN(num_features=1, hidden_dim=32).to(device)
	model.apply(initWeights)
	# Make the model predict the regression outputs and save the results
	predictions = test(model, torch_dataset_loader)
	writePickle(predictions, f"{args.path}/teacher_outputs/{filename.rstrip('.pkl')}_{int(time.time())}.pkl")


if __name__ == '__main__':
	main()
