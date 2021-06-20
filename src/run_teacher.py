import sys
import time
import torch
import inspect
import argparse
import importlib
import numpy as np
import networkx as nx

from tqdm import tqdm
from functools import partial

from torch import nn
from torch_geometric.data import DataLoader

from utils.convert import fromNetworkx2Torch
from utils.io import readPickle, writePickle, parameterizedKeepOrderAction



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
    parser.add_argument(
        '--verbose', '-v', type=bool, default=True, 
        help='Whether to print the outputs through the terminal.')
    parser.add_argument(
        '--save_file_destination', type=bool, default=False,
        help='Whether to save the file path destination into a temporary file for later pipelined processing.')
    ##########
    # Setting specific arguments
    parser.add_argument(
        '--setting', type=str, required=True, choices=['regression', 'classification'], 
        help='Setting used for producing outputs [classification, regression].')
    parser.add_argument(
        '--classes', type=int, #required='--setting classification' in ' '.join(sys.argv),
        help='Number of classes for the classification setting.')
    ##########
    # Network weight initialization specific arguments
    # parser.add_argument(
    #     '--init_weights', type=str, default='uniform', choices=['uniform'],
    #     help='Weight initialization scheme for the chosen network architecture.')
    parser.add_argument(
        '--bias', type=float, action=parameterizedKeepOrderAction('init_kwargs'),
        help='Whether to include some specific bias to the linear layers of the network.')
    parser.add_argument(
        '--lower_bound', type=float, action=parameterizedKeepOrderAction('init_kwargs'),
        help='Lower bound of the uniform random initialization.')
    parser.add_argument(
        '--upper_bound', type=float, action=parameterizedKeepOrderAction('init_kwargs'),
        help='Upper bound of the uniform random initialization.')
    ##########
    # Model specific arguments
    model_subparser = parser.add_subparsers(dest='model')
    model_subparser.required = True
    ###
    GIN = model_subparser.add_parser('GIN', help='GIN model specific parser.')
    GIN.add_argument(
        '--num_features', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of input features per node.')
    GIN.add_argument(
        '--hidden_dim', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of hidden neurons per hidden linear layer.')
    GIN.add_argument(
        '--residual', type=bool, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Whether to add residual connections in the network.')
    GIN.add_argument(
        '--jk', type=bool, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Whether to add jumping knowledge in the network.')
    # TBD add more models and more parameterizations
    return parser.parse_args()


def resolveParameters(f, kwargs):
    '''Auxiliary function to resolve the given parameters against the function defaults.'''
    default_f_kwargs = {
        k: v.default for k, v in inspect.signature(f).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    resolved_kwargs = {
        k: kwargs[k] if k in kwargs else v for k, v in default_f_kwargs.items()
    }
    return resolved_kwargs


def main():
    args = readArguments()
    # Read the dataset and convert it to torch_geometric.data
    networkx_dataset = readPickle(args.dataset_filename)
    torch_dataset = fromNetworkx2Torch(networkx_dataset, initial_relabeling=args.initial_relabeling)
    torch_dataset_loader = DataLoader(torch_dataset, batch_size=1)
    # Import the model
    module = importlib.import_module(f'models.{args.model}.{args.setting}')
    # Resolve function parameters
    init_kwargs = {k: v for k, v in args.init_kwargs} if 'init_kwargs' in vars(args) else {}
    init_kwargs = resolveParameters(module.initWeights, init_kwargs)
    model_kwargs = {k: v for k, v in args.model_kwargs} if 'model_kwargs' in vars(args) else {}
    if args.setting == 'classification' and args.classes:
        model_kwargs['classes'] = args.classes
    model_kwargs = resolveParameters(module.Net.__init__, model_kwargs)
    teacher_outputs_filename = \
        f"{'/'.join(args.dataset_filename.split('/')[:-1])}/teacher_outputs/{args.setting}/{args.model}/" \
        f"{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in model_kwargs.items()])}__" \
        f"{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in init_kwargs.items()])}__{int(time.time())}/teacher_outputs.pkl"
    # Init the model
    model = module.Net(**model_kwargs).to(device)
    model.apply(partial(module.initWeights, **init_kwargs))
    # Make the model predict the regression outputs and save the results
    teacher_outputs = module.test(model, torch_dataset_loader, device)
    if args.verbose:
        print()
        print('Teacher outputs:')
        print('-' * 30)
        print(teacher_outputs)
    writePickle(teacher_outputs, filename=teacher_outputs_filename)
    return teacher_outputs_filename if args.save_file_destination else ''


if __name__ == '__main__':
    tmp_saved_filename = main()
    # sys.exit(predictions_filename)
    # Workaround to pass variable name to bash script...
    if tmp_saved_filename:
        with open('tmp_saved_filename.txt', 'w') as f:
            f.write(tmp_saved_filename)