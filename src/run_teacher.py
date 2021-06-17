import sys
import time
import torch
import argparse
import importlib
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch import nn
from torch_geometric.data import DataLoader

from utils.io import readPickle, writePickle
from utils.convert import *



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--dataset_filename', type=str, required=True, 
    	help='Full relative path to the networkx dataset.')
    parser.add_argument(
        '--initial_relabeling', type=str, required=True, choices=['ones', 'degrees'],
        help='Type of labeling to be used in the case that there aren\'t any available. Available choices are [ones, degrees].')
    ###
   	# Setting specific arguments
    parser.add_argument(
    	'--setting', type=str, required=True, choices=['regression', 'classification'], 
    	help='Setting used for producing outputs [classification, regression].')
   	parser.add_argument(
   		'--num_classes', type=int, required='--setting classification' in ' '.join(sys.argv),
   		help='Number of classes for the classification setting.')
    ###
    # Model specific arguments
    parser.add_argument(
    	'--model', '-m', type=str, default='GIN', help='Graph Neural Network architecture to be used.')
   	# TO BE CONTINUED
    return parser.parse_args()


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(f'{args.path}/raw')
	# Read the dataset and convert it to torch_geometric.data
	networkx_dataset = readPickle(args.dataset_filename)
	#
	torch_dataset = fromNetworkx2Torch(networkx_dataset, initial_relabeling=initial_relabeling)
	torch_dataset_loader = DataLoader(torch_dataset, batch_size=1)
	# Import the model
	module = importlib.import_module(f'models.{args.model}.{args.setting}')
	# Init the model
	model = module.Net(num_features=1, hidden_dim=32).to(device)
	model.apply(module.initWeights)
	# Make the model predict the regression outputs and save the results
	predictions = module.test(model, torch_dataset_loader, device)
	#writePickle(predictions, filename=f"{args.path}/teacher_outputs/{filename.rstrip('.pkl')}_{int(time.time())}.pkl")


if __name__ == '__main__':
	main()
