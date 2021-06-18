import sys
import time
import torch
import argparse
import importlib
import numpy as np
import networkx as nx

from tqdm import tqdm
from functools import partial

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
        '--model', '-m', type=str, required=True, choices=['GIN', 'GCN', 'SIGN', 'SGC', 'ChebNet'], 
        help='Graph Neural Network architecture to be used.')
    parser.add_argument(
        '--model_kwargs', nargs='+',
        help='Specific model parameters depending on the chosen architecture.')
    return parser.parse_args()


def parseModelKwargs(unparsed_args):
    '''Auxiliary function to parse model keyword arguments.'''
    model_kwargs, init_kwargs = {}, {}
    for element in unparsed_args:
        if element.startswith('_'):
            k = element.lstrip('_')
            model_kwargs[k] = []
        else:
            model_kwargs[k] = int(element) if element.isnumeric() else element
    return model_kwargs, init_kwargs


def main():
    args = readArguments()
    model_kwargs, _ = parseModelKwargs(args.model_kwargs if args.model_kwargs else [])
    # Read the dataset and convert it to torch_geometric.data
    networkx_dataset = readPickle(args.dataset_filename)
    torch_dataset = fromNetworkx2Torch(networkx_dataset, initial_relabeling=args.initial_relabeling)
    torch_dataset_loader = DataLoader(torch_dataset, batch_size=1)
    # Import the model
    module = importlib.import_module(f'models.{args.model}.{args.setting}')
    # Init the model
    model = module.Net(**model_kwargs).to(device)
    model.apply(partial(
        module.initWeights, bias=0, lower_bound=-0.3, upper_bound=0.3))
    # Make the model predict the regression outputs and save the results
    predictions = module.test(model, torch_dataset_loader, device)
    print(predictions)
    #writePickle(predictions, filename=f"{args.path}/teacher_outputs/{filename.rstrip('.pkl')}_{int(time.time())}.pkl")


if __name__ == '__main__':
    main()
