import torch
import argparse
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch import nn
from torch_geometric.data import DataLoader

from utils.io import *
from utils.data import *
from utils.stats import *
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


def train(model, optimizer, loader, epochs=10):
	'''
	Student train function.

	Parameters:
		- model: (models.GINRegressor.GIN)
		- optimizer: (torch.optim) Optimizer for training.
		- loader: (torch_geometric.data.dataloader.DataLoader) Torch data loader for training.
		- epochs: (int) Number of epochs for training.

	Returns:
		- None
	'''
	print()
	print('-' * 30)
	print('Init training...')
	model.train()
	loss = nn.MSELoss()
	for epoch in range(epochs):
		print(f'Epoch {epoch + 1}')
		for data in tqdm(loader):
			data = data.to(device)
			optimizer.zero_grad()
			output = loss(model(data), data.y)
			output.backward()
			optimizer.step()
	return None


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
	print('Evaluating on test data...')
	model.eval()
	predictions = []
	for data in tqdm(loader):
		data = data.to(device)
		output = model(data)
		predictions.append(output.detach().numpy().reshape(-1))
	return predictions


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(f'{args.path}/raw')
	# Read the dataset and convert it to torch_geometric.data
	networkx_dataset = readPickle(f'{args.path}/raw/{filename}')
	torch_dataset = fromNetworkx(networkx_dataset, add_degree=True)
	# Read the teacher outputs of the dataset and split the data
	regression_outputs_filename = getLatestVersion(
		f'{args.path}/teacher_outputs', filename=filename.rstrip('.pkl'))
	regression_outputs = readPickle(f'{args.path}/teacher_outputs/{regression_outputs_filename}')
	torch_dataset = addLabels(torch_dataset, regression_outputs)
	X_train, X_test = splitData(torch_dataset)
	train_loader = DataLoader(X_train, batch_size=1)
	test_loader = DataLoader(X_test, batch_size=1)
	# Init the model
	model = GIN(num_features=1, hidden_dim=32).to(device)
	model.apply(initWeights)
	# Init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	# Train the model
	train(model, optimizer, train_loader, epochs=1)
	# Predict on unseen data
	predictions = test(model, test_loader)
	# Evaluate the performance
	total_error, avg_G_error, avg_n_error = evaluatePerformance(predictions, [G.y.numpy() for G in X_test])
	print(total_error, avg_G_error, avg_n_error)


if __name__ == '__main__':
	main()









	