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

from utils.training import *
from utils.convert import fromNetworkx2Torch
from utils.io import readPickle, writePickle, parameterizedKeepOrderAction, booleanString



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
        '--verbose', type=booleanString, default=False, 
        help='Whether to print the outputs through the terminal.')
    parser.add_argument(
        '--save_file_destination', type=booleanString, default=False,
        help='Whether to save the file path destination into a temporary file for later pipelined processing.')
    parser.add_argument(
        '--num_iterations', type=int, default=1,
        help='Number of teacher outputs to produce with the same configuration.')
    ##########
    # Setting specific arguments
    parser.add_argument(
        '--setting', type=str, required=True, choices=['regression', 'classification'], 
        help='Setting used for producing outputs [classification, regression].')
    parser.add_argument(
        '--classes', type=int, required='--setting classification' in ' '.join(sys.argv),
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
        '--blocks', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of GIN blocks to include in the model.')
    GIN.add_argument(
        '--residual', type=booleanString, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Whether to add residual connections in the network.')
    GIN.add_argument(
        '--jk', type=booleanString, action=parameterizedKeepOrderAction('model_kwargs'),
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
    Net = getattr(importlib.import_module(f'models.{args.model}'), 'Net')
    # Resolve initialization parameters
    init_kwargs = {k: v for k, v in args.init_kwargs} if 'init_kwargs' in vars(args) else {}
    init_kwargs = resolveParameters(initWeights, init_kwargs)
    # Resolve model parameters and teacher filename
    teacher_outputs_filename_prefix = \
        f"{'/'.join(args.dataset_filename.split('/')[:-1])}/teacher_outputs/{args.setting}"
    model_kwargs = {k: v for k, v in args.model_kwargs} if 'model_kwargs' in vars(args) else {}
    if args.setting == 'classification' and args.classes:
        teacher_outputs_filename_prefix = f"{teacher_outputs_filename_prefix}/{args.classes}"
        model_kwargs['num_outputs'] = args.classes
    model_kwargs = resolveParameters(Net.__init__, model_kwargs)
    teacher_outputs_filename_prefix = \
        f"{teacher_outputs_filename_prefix}/{args.model}/" \
        f"{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in model_kwargs.items() if k not in ['num_features', 'num_outputs']])}__" \
        f"{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in init_kwargs.items()])}"   
    for _ in tqdm(range(args.num_iterations)):
        teacher_outputs_filename_prefix_ = f"{teacher_outputs_filename_prefix}/{int(time.time() * 1000)}"
        teacher_outputs_filename = f"{teacher_outputs_filename_prefix_}/teacher_outputs.pkl"
        teacher_outputs_filename_model = f"{teacher_outputs_filename_prefix_}/model.pt"
        # Init the model
        model = Net(**model_kwargs).to(device)
        model.apply(partial(initWeights, **init_kwargs))
        # Make the model predict the regression outputs and save the results
        if args.setting == 'regression':
            teacher_outputs = test_regression(model, torch_dataset_loader, device)
        else: #elif args.setting == 'classification':
            teacher_outputs = test_classification(model, torch_dataset_loader, device)
        if args.verbose:
            print()
            print('Teacher outputs:')
            print('-' * 30)
            print(teacher_outputs)
        # Save the teacher outputs and the model
        writePickle(teacher_outputs, filename=teacher_outputs_filename)
        torch.save(model.state_dict(), teacher_outputs_filename_model)
    return teacher_outputs_filename if args.save_file_destination else ''


if __name__ == '__main__':
    tmp_saved_filename = main()
    # sys.exit(predictions_filename)
    # Workaround to pass variable name to bash script...
    if tmp_saved_filename:
        with open('tmp_saved_filename.txt', 'w') as f:
            f.write(tmp_saved_filename)
