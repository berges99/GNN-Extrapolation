import torch
import argparse
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch import nn
from torch_geometric.data import DataLoader

from utils.io import *
from utils.data import *
from utils.convert import *
from models.GINRegressor import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--filename', '-f', type=str, default='', help='Filename of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/erdos_renyi', help='Default path to the data.')
    return parser.parse_args()


def initWeights(m):
	'''Auxiliary function that applies a uniform distribution to the weights and a bias=0.'''
	if type(m) == nn.Linear:
		m.weight.data.uniform_(-0.3, 0.3)
		m.bias.data.fill_(0)


def train(model, optimizer, loader, epochs=10):
	''''''
	model.train()
	loss = nn.MSELoss()
	for epoch in range(epochs):
		for data in tqdm(loader):
			data = data.to(device)
			optimizer.zero_grad()
			output = loss(model(data), data.y)
			output.backward()
			optimizer.step()
	return None


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(args.path)
	# Read the dataset and convert it to torch_geometric.data
	networkx_dataset = readPickle(f'{args.path}/{filename}')
	torch_dataset = fromNetworkx(networkx_dataset, add_degree=True)
	# Read the teacher outputs of the dataset and split the data
	regression_outputs = readPickle(f"{args.path}/{filename.rstrip('.pkl')}_teacher.pkl")
	torch_dataset = addLabels(torch_dataset, regression_outputs)
	#X_train, X_test = splitData(torch_dataset)
	train_loader = DataLoader(torch_dataset, batch_size=1)
	#test_loader = DataLoader(list(X_test), batch_size=1)
	# Init the model
	model = GIN(num_features=1, hidden_dim=32).to(device)
	model.apply(initWeights)
	# Init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	# Train the model
	train(model, optimizer, train_loader, epochs=1)


if __name__ == '__main__':
	main()