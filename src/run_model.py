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

from utils.convert import fromNetworkx2Torch, addLabels
from utils.io import readPickle, writePickle, parameterizedKeepOrderAction



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_filename', type=str, # no longer required (baseline)
        help='Full relative path to the networkx dataset.')
    parser.add_argument(
        '--teacher_outputs_filename', type=str, required=True, 
        help='Full relative path to the teacher outputs of the given dataset.')
    parser.add_argument(
        '--initial_relabeling', type=str, choices=['ones', 'degrees'], # no longer required (baseline)
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
    # Training specific arguments
    parser.add_argument(
        '--epochs', type=int, default=3,
        help='Number of epochs of training for the chosen model.')
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
    Baseline = model_subparser.add_parser('Baseline', help='Baseline model specific parser.')
    Baseline.add_argument(
        '--smoothing', type=str, required='Baseline' in sys.argv, 
        choices=['none', 'knn', 'heat_kernel', 'approx_pagerank'],
        help='Type of smoothing to apply to the baseline model.')
    Baseline.add_argument(
        '--dist_matrix_filename', type=str, required='Baseline' in sys.argv,
        help='Full relative path to the node representations to be used by the baseline method.')
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
    # Read the teacher outputs of the dataset
    teacher_outputs = readPickle(args.teacher_outputs_filename)
    student_outputs = []
    # Import the model
    module = importlib.import_module(f'models.{args.model}.{args.setting}')
    # Storage filename prefix
    student_outputs_filename = \
        f"{'/'.join(args.teacher_outputs_filename.split('/')[:-1])}/student_outputs/{args.model}"
    if args.model == 'Baseline':
        # Resolve storage filename
        student_outputs_filename = \
            f"{student_outputs_filename}/" \
            f"{('/'.join(args.dist_matrix_filename.split('/')[-6:-3]) + '/' + '/'.join(args.dist_matrix_filename.split('/')[-2:])).replace('/', '__').rstrip('.pkl')}/" \
            f"{args.smoothing}.pkl"
        # Read the specified node representations
        node_representations, dist_matrix = readPickle(args.dist_matrix_filename)
        # Compute number of nodes per graph (to handle multiple sized graphs in the future)
        node_representations_idxs = np.array([len(G) for G in node_representations], dtype=int)
        teacher_outputs_flatten = np.array([item for sublist in teacher_outputs for item in sublist])
        node_representations_flatten = np.array([item for sublist in node_representations for item in sublist])
        _, student_outputs = module.baseline(
            node_representations_flatten, node_representations_idxs, teacher_outputs_flatten, dist_matrix,
            train_idxs=list(range(len(node_representations))), test_idxs=None, smoothing=args.smoothing
        )
    # If the model is a GNN
    else: #elif args.model != 'Baseline':
        # Resolve the model arguments
        init_kwargs = {k: v for k, v in args.init_kwargs} if 'init_kwargs' in vars(args) else {}
        init_kwargs = resolveParameters(module.initWeights, init_kwargs)
        model_kwargs = {k: v for k, v in args.model_kwargs} if 'model_kwargs' in vars(args) else {}
        if args.setting == 'classification' and args.classes:
            model_kwargs['classes'] = args.classes
        model_kwargs = resolveParameters(module.Net.__init__, model_kwargs)
        # Resolve storage filename
        student_outputs_filename = \
            f"{student_outputs_filename}/{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in model_kwargs.items()])}__" \
            f"{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in init_kwargs.items()])}__{int(time.time())}.pkl"
        # Read the dataset and convert it to torch_geometric.data
        networkx_dataset = readPickle(args.dataset_filename)
        torch_dataset = fromNetworkx2Torch(networkx_dataset, initial_relabeling=args.initial_relabeling)
        torch_dataset = addLabels(torch_dataset, teacher_outputs)
        ###
        # X_train, X_test = splitData(torch_dataset)
        # train_loader = DataLoader(X_train, batch_size=1)
        # test_loader = DataLoader(X_test, batch_size=1)
        ###
        torch_dataset_loader = DataLoader(torch_dataset, batch_size=1)
        # Init the model
        model = module.Net(**model_kwargs).to(device)
        model.apply(partial(module.initWeights, **init_kwargs))
        # Train during the specified amount of epochs
        student_outputs.append(module.test(model, torch_dataset_loader, device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(args.epochs):
            module.train(model, optimizer, torch_dataset_loader, device)
            student_outputs.append(module.test(model, torch_dataset_loader, device))
    ###
    writePickle(student_outputs, filename=student_outputs_filename)
    if args.verbose:
        print()
        print('Student outputs:')
        print('-' * 30)
        print(student_outputs)
    return student_outputs_filename if args.save_file_destination else ''


if __name__ == '__main__':
    tmp_saved_filename = main()
    # sys.exit(predictions_filename)
    # Workaround to pass variable name to bash script...
    if tmp_saved_filename:
        with open('tmp_saved_filename.txt', 'w') as f:
            f.write(tmp_saved_filename)