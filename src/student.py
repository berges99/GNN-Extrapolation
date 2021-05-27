import torch
import argparse
import importlib
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch import nn
from torch_geometric.data import DataLoader

from utils.io import *
from utils.data import *
from utils.stats import *
from utils.convert import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--filename', '-f', type=str, default='', help='Filename of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/preferential_attachment', help='Default path to the data.')
    parser.add_argument(
    	'--model', '-m', type=str, default='GIN', help='Graph Neural Network architecture to be used.')
    parser.add_argument(
    	'--setting', '-s', type=str, default='regression', help='Setting used for producing outputs [classification, regression].')
    return parser.parse_args()


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(f'{args.path}/raw')
	# Read the dataset and convert it to torch_geometric.data
	networkx_dataset = readPickle(f'{args.path}/raw/{filename}')
	torch_dataset = fromNetworkx2Torch(networkx_dataset, add_degree=True)
	# Read the teacher outputs of the dataset and split the data
	regression_outputs_filename = getLatestVersion(
		f'{args.path}/teacher_outputs', filename=filename.rstrip('.pkl'))
	regression_outputs = readPickle(f'{args.path}/teacher_outputs/{regression_outputs_filename}')
	torch_dataset = addLabels(torch_dataset, regression_outputs)
	X_train, X_test = splitData(torch_dataset)
	train_loader = DataLoader(X_train, batch_size=1)
	test_loader = DataLoader(X_test, batch_size=1)
	# Import the model
	module = importlib.import_module(f'models.{args.model}.{args.setting}')
	# Init the model
	model = module.Net(num_features=1, hidden_dim=32).to(device)
	model.apply(module.initWeights)
	# Init optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	# Train the model
	module.train(model, optimizer, train_loader, epochs=1, device=device)
	# Predict on unseen data
	predictions = module.test(model, test_loader, device)
	print(predictions)
	# # Evaluate the performance
	# total_error, avg_G_error, avg_n_error = evaluatePerformance(predictions, [G.y.numpy() for G in X_test])
	# print(total_error, avg_G_error, avg_n_error)


if __name__ == '__main__':
	main()

	