import torch
import argparse
import numpy as np
import networkx as nx

from torch import nn
from torch_geometric.data import DataLoader

from models.GIN import GIN
from utils.io import readPickleNetworkx, fromNetworkx


from collections import Counter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--filename', '-f', type=str, required=True, help='Relative path to the dataset to be used.')
    return parser.parse_args()


def initWeights(m):
	# Apply a uniform distribution to the weights and a bias=0
	if type(m) == nn.Linear:
		m.weight.data.uniform_(-1, 1)
		m.bias.data.fill_(0)


def test(model, loader):
	''''''
	model.eval()
	#
	for data in loader:
		data = data.to(device)
		output = model(data)
		pred = output.max(dim=1)[1]
		print(output)
		print(pred)
		print(Counter(np.array(pred)))


def main():
	args = readArguments()
	# Read the dataset and convert it to torch_geometric.data
	networkx_dataset = readPickleNetworkx(args.filename)
	torch_dataset = fromNetworkx(networkx_dataset, add_degree=True)
	torch_dataset_loader = DataLoader(torch_dataset, batch_size=1)
	# Init the model
	model = GIN(num_features=1, hidden_dim=32, num_classes=4).to(device)
	model.apply(initWeights)

	test(model, torch_dataset_loader)


if __name__ == '__main__':
	main()