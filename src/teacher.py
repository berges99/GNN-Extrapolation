import time
import torch
import argparse
import importlib
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch import nn
from torch_geometric.data import DataLoader

from utils.io import *
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
	torch_dataset_loader = DataLoader(torch_dataset, batch_size=1)
	# Import the model
	module = importlib.import_module(f'models.{args.model}.{args.setting}')
	# Init the model
	model = module.Net(num_features=1, hidden_dim=32).to(device)
	model.apply(module.initWeights)
	# Make the model predict the regression outputs and save the results
	predictions = module.test(model, torch_dataset_loader, device)
	writePickle(predictions, f"{args.path}/teacher_outputs/{filename.rstrip('.pkl')}_{int(time.time())}.pkl")


if __name__ == '__main__':
	main()
