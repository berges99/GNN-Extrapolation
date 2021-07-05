import os
import re
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
from collections import defaultdict

from torch import nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from utils.convert import fromNetworkx2Torch, addLabels
from utils.io import readPickle, writePickle, parameterizedKeepOrderAction, booleanString
from utils.training import train_regression, train_classification, test_regression, test_classification



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    ##########
    # Input data to the "student"
    parser.add_argument(
        '--dataset_filename', type=str,
        help='Full relative path to the networkx dataset.')
    parser.add_argument(
        '--initial_relabeling', type=str, choices=['ones', 'degrees'],
        help='Type of labeling to be used in the case that there aren\'t any available.')
    ###
    parser.add_argument(
        '--teacher_outputs_filename', type=str, required=True, 
        help='Full relative path to the teacher outputs of the given dataset.')
    ##########
    # Miscellaneous arguments
    parser.add_argument(
        '--verbose', type=booleanString, default=False, 
        help='Whether to print the outputs through the terminal.')
    parser.add_argument(
        '--save_file_destination', type=booleanString, default=False,
        help='Whether to save the file path destination into a temporary file for later pipelined processing.')
    ###
    parser.add_argument(
        '--num_iterations', type=int, default=10,
        help='Number of student outputs to produce with the same configuration.')
    ##########
    # Training specific arguments
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of epochs of training for the chosen model.')
    parser.add_argument(
        '--lr', type=float, default=2e-04,
        help='Adam learning rate.')
    ##########
    # Network weight initialization specific arguments
    parser.add_argument(
        '--init', type=str, default='uniform', choices=['uniform', 'xavier'],
        help='Type of initialization for the network.')
    parser.add_argument(
        '--bias', type=float, action=parameterizedKeepOrderAction('init_kwargs'),
        help='Whether to include some specific bias to the linear layers of the network.')
    ###
    parser.add_argument(
        '--lower_bound', type=float, action=parameterizedKeepOrderAction('init_kwargs'),
        help='Lower bound of the uniform random initialization.')
    parser.add_argument(
        '--upper_bound', type=float, action=parameterizedKeepOrderAction('init_kwargs'),
        help='Upper bound of the uniform random initialization.')
    ###
    parser.add_argument(
        '--gain', type=float, action=parameterizedKeepOrderAction('init_kwargs'),
        help='Optional scaling factor for the Xavier initialization.')
    ##########
    # Model specific arguments
    model_subparser = parser.add_subparsers(dest='model')
    model_subparser.required = True
    ###
    Baseline = model_subparser.add_parser('Baseline', help='Baseline model specific parser.')
    Baseline.add_argument(
        '--dist_matrix_filename', type=str, required='Baseline' in sys.argv,
        help='Full relative path to the node representations to be used by the baseline method.')
    Baseline.add_argument(
        '--method', type=str, required='Baseline' in sys.argv, choices=['baseline', 'knn'],
        help='Type of smoothing to apply to the baseline model.')
    ###
    Baseline.add_argument(
        '--smoothing', type=str, choices=['tikhonov', 'heat_kernel', 'pagerank_kernel'],
        help='Full relative path to the node representations to be used by the baseline method.')
    Baseline.add_argument(
        '--gamma', type=float, required='--smoothing tikhonov' in ' '.join(sys.argv),
        action=parameterizedKeepOrderAction('smoothing_kwargs'),
        help='Gamma parameter for the Tikhonov regularization.')
    Baseline.add_argument(
        '--tau', type=float, required='--smoothing heat_kernel' in ' '.join(sys.argv),
        action=parameterizedKeepOrderAction('smoothing_kwargs'),
        help='Tau parameter for the heat kernel.')
    # TBD Implement the approximate pagerank kernel
    ##########
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
    ###
    GCN = model_subparser.add_parser('GCN', help='GCN model specific parser.')
    GCN.add_argument(
        '--num_features', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of input features per node.')
    GCN.add_argument(
        '--hidden_dim', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of hidden neurons per hidden linear layer.')
    GCN.add_argument(
        '--blocks', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of GCN blocks to include in the model.')
    GCN.add_argument(
        '--residual', type=booleanString, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Whether to add residual connections in the network.')
    GCN.add_argument(
        '--jk', type=booleanString, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Whether to add jumping knowledge in the network.')
    ###
    SIGN = model_subparser.add_parser('SIGN', help='SIGN model specific parser.')
    SIGN.add_argument(
        '--num_features', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of input features per node.')
    SIGN.add_argument(
        '--hidden_dim', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of hidden neurons per hidden linear layer.')
    SIGN.add_argument(
        '--K', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of hops/layer.')
    SIGN.add_argument(
        '--pre_linear', type=booleanString, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Whether to individually apply linear projections to each of the input channels.')
    ###
    SGC = model_subparser.add_parser('SGC', help='SGC model specific parser.')
    SGC.add_argument(
        '--num_features', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of input features per node.')
    SGC.add_argument(
        '--hidden_dim', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of hidden neurons per hidden linear layer.')
    SGC.add_argument(
        '--K', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of hops/layer.')
    ###
    ChebNet = model_subparser.add_parser('ChebNet', help='ChebNet model specific parser.')
    ChebNet.add_argument(
        '--num_features', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of input features per node.')
    ChebNet.add_argument(
        '--hidden_dim', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of hidden neurons per hidden linear layer.')
    ChebNet.add_argument(
        '--blocks', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Number of ChebNet blocks to include in the model.')
    ChebNet.add_argument(
        '--K', type=int, action=parameterizedKeepOrderAction('model_kwargs'),
        help='Chebyshev filter size K (order of the polynomial).')
    ChebNet.add_argument(
        '--normalization', type=str, action=parameterizedKeepOrderAction('model_kwargs'),
        choices=['sym', 'rw'], help='The normalization scheme for the graph Laplacian.')
    ###
    return parser.parse_args()


def resolveParameters(f, kwargs):
    '''Auxiliary function to resolve the given parameters against the function defaults.'''
    default_f_kwargs = {
        k: v.default for k, v in inspect.signature(f).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    kwargs_ = {}
    full_f_names = [k.split('_') for k in default_f_kwargs.keys()]
    for k, v in kwargs.items():
        for name in full_f_names:
            if k == name[0] or k == '_'.join(name):
                kwargs_['_'.join(name)] = v
    resolved_kwargs = {
        k: kwargs_[k] if k in kwargs_ else v for k, v in default_f_kwargs.items()
    }
    return resolved_kwargs


def resolveTeacherOutputsFilenames(path):
    '''Auxiliary function that returns an iterable with all the involved teacher outputs.'''
    if 'teacher_outputs.pkl' in os.listdir(path):
        teacher_outputs_filenames = [path]
    else:
        teacher_outputs_filenames = [
            f"{path}/{t_i}/teacher_outputs.pkl" for t_i in os.listdir(path)
            if all(c.isdigit() for c in t_i[-13:])
        ]
    return teacher_outputs_filenames


def main():
    args = readArguments()
    # Resolve regression vs classification setting
    args.setting = 'regression' if 'regression' in args.teacher_outputs_filename else 'classification'
    if args.setting == 'classification':
        args.classes = int(args.teacher_outputs_filename.split('/classification/')[-1].split('/')[0])
    if args.model == 'Baseline':
        # Import the model
        Baseline = getattr(importlib.import_module(f'models.{args.model}'), 'Baseline')
        # Resolve smoothing keyword arguments
        smoothing_kwargs = {k: v for k, v in args.smoothing_kwargs} if 'smoothing_kwargs' in vars(args) else {}
        # Resolve storage filename
        student_outputs_filename_suffix = \
            f"{'/'.join(args.dist_matrix_filename.split('/')[-8:]).replace('/', '__').rstrip('.npz')}" \
            f"__{args.method}_s{str(args.smoothing).capitalize()}"
        if smoothing_kwargs:
            student_outputs_filename_suffix = \
                f"{student_outputs_filename_suffix}_{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in smoothing_kwargs.items()])}"
        student_outputs_filename_suffix = f"{student_outputs_filename_suffix}.pkl"
        # Read the distance matrices. Example of path:
        # ../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sMaxdegree/train/full64.npz
        dist_matrix_train = np.load(args.dist_matrix_filename)
        idxs = dist_matrix_train['idxs'] if 'nystrom' in args.dist_matrix_filename.split('/')[-1] else None
        dist_matrix_train = dist_matrix_train['dist_matrix']
        # Read the test and extrapolation matrices (ensure to take the (n x m) matrix!)
        dist_matrix_test = np.load(
            f"{'/'.join(args.dist_matrix_filename.split('/')[:-2])}/test/full64_test.npz")['dist_matrix']
        dist_matrix_extrapolation = np.load(
            f"{'/'.join(args.dist_matrix_filename.split('/')[:-2])}/extrapolation/full64_test.npz")['dist_matrix']
        # Read the teacher outputs of the dataset (iterate through folder if necessary)
        for teacher_outputs_filename in tqdm(resolveTeacherOutputsFilenames(args.teacher_outputs_filename)):
            # Resolve storage filename
            student_outputs_filename = \
                f"{'/'.join(teacher_outputs_filename.split('/')[:-1])}/student_outputs/{args.model}/{student_outputs_filename_suffix}"
            # Read the teacher outputs (train, test, extrapolation)
            teacher_outputs_train = readPickle(teacher_outputs_filename)['train']
            teacher_outputs_train_flatten = np.array([item for sublist in teacher_outputs_train for item in sublist])
            # Predict with the baseline (predict on all the training set) -> list(range(len(node_representations))) 
            # (train, test, extrapolation)
            student_outputs = {}
            student_outputs['train'] = [Baseline(
                teacher_outputs_train_flatten, dist_matrix_train, idxs=idxs, 
                num_outputs=1 if args.setting == 'regression' else args.classes,
                method=args.method, smoothing=None, **smoothing_kwargs #smoothing=args.smoothing
            )]
            student_outputs['test'] = [Baseline(
                teacher_outputs_train_flatten, dist_matrix_test, idxs=None, 
                num_outputs=1 if args.setting == 'regression' else args.classes,
                method=args.method, smoothing=None, **smoothing_kwargs #smoothing=args.smoothing
            )]
            student_outputs['extrapolation'] = [Baseline(
                teacher_outputs_train_flatten, dist_matrix_extrapolation, idxs=None, 
                num_outputs=1 if args.setting == 'regression' else args.classes,
                method=args.method, smoothing=None, **smoothing_kwargs #smoothing=args.smoothing
            )]
            if args.verbose:
                print()
                print('Student outputs:')
                print('-' * 30)
                print(student_outputs)
                print(student_outputs_filename)
            writePickle(student_outputs, filename=student_outputs_filename)
    # If the model is a GNN
    else: #elif args.model != 'Baseline':
        # Read the dataset and convert it to torch_geometric.data
        networkx_dataset = readPickle(args.dataset_filename)
        # Convert it into pytorch data and split (train, test, extrapolation)
        torch_dataset_train = fromNetworkx2Torch(
            networkx_dataset['train'], initial_relabeling=args.initial_relabeling)
        torch_dataset_test = fromNetworkx2Torch(
            networkx_dataset['test'], initial_relabeling=args.initial_relabeling)
        torch_dataset_extrapolation = fromNetworkx2Torch(
            networkx_dataset['extrapolation'], initial_relabeling=args.initial_relabeling)
        # Resolve model and init kwargs and import the necessary module
        initWeights = getattr(importlib.import_module('utils.training'), f'initWeights{args.init.capitalize()}')
        init_kwargs = {k: v for k, v in args.init_kwargs} if 'init_kwargs' in vars(args) else {}
        init_kwargs = resolveParameters(initWeights, init_kwargs)
        model_kwargs = {k: v for k, v in args.model_kwargs} if 'model_kwargs' in vars(args) else {}
        if args.setting == 'classification' and args.classes:
            model_kwargs['num_outputs'] = args.classes
        # Import the model
        Net = getattr(importlib.import_module(f'models.{args.model}'), 'Net')
        model_kwargs = resolveParameters(Net.__init__, model_kwargs)
        # Read the teacher outputs of the dataset (iterate through folder if necessary)
        for teacher_outputs_filename in tqdm(resolveTeacherOutputsFilenames(args.teacher_outputs_filename)):
            # Resolve storage filename
            student_outputs_filename_prefix = \
                f"{'/'.join(teacher_outputs_filename.split('/')[:-1])}/student_outputs/{args.model}/" \
                f"{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in model_kwargs.items() if k not in ['num_features', 'num_outputs']])}__" \
                f"init{args.init.capitalize()}_{'_'.join([k.split('_')[0] + str(v).capitalize() for k, v in init_kwargs.items()])}"
            # Add the labels of the training data (no need to read the labels of test & extrapolation data)
            teacher_outputs_train = readPickle(teacher_outputs_filename)['train']
            torch_dataset_train = addLabels(torch_dataset_train, teacher_outputs_train)
            # Apply special transformation if necessary (SIGN & ChebNet)
            if args.model in ['SIGN', 'ChebNet']:
                if args.model == 'SIGN':
                    transform = T.Compose([T.NormalizeFeatures(), T.SIGN(model_kwargs['K'])])
                elif args.model == 'ChebNet':
                    # Init transform that obtains the highest eigenvalue of the graph Laplacian given by 
                    # torch_geometric.utils.get_laplacian()
                    transform = T.Compose([T.LaplacianLambdaMax(
                        normalization=model_kwargs['normalization'], is_undirected=True)])
                torch_dataset_train = [transform(G) for G in torch_dataset_train]
                torch_dataset_test = [transform(G) for G in torch_dataset_test]
                torch_dataset_extrapolation = [transform(G) for G in torch_dataset_extrapolation]
            # Init the data loaders
            torch_dataset_train_loader = DataLoader(torch_dataset_train, batch_size=1)
            torch_dataset_test_loader = DataLoader(torch_dataset_test, batch_size=1)
            torch_dataset_extrapolation_loader = DataLoader(torch_dataset_extrapolation, batch_size=1) 
            # Produce as many results with as many random initializations as indicated
            for _ in range(args.num_iterations):
                # Init the model
                model = Net(**model_kwargs).to(device)
                model.apply(partial(initWeights, **init_kwargs))
                # Train during the specified amount of epochs
                student_outputs = defaultdict(list)
                if args.setting == 'regression':
                    student_outputs['train'].append(test_regression(model, torch_dataset_train_loader, device))
                    student_outputs['test'].append(test_regression(model, torch_dataset_test_loader, device))
                    student_outputs['extrapolation'].append(test_regression(model, torch_dataset_extrapolation_loader, device))
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    for _ in range(args.epochs):
                        train_regression(model, optimizer, torch_dataset_train_loader, device)
                        student_outputs['train'].append(test_regression(model, torch_dataset_train_loader, device))
                        student_outputs['test'].append(test_regression(model, torch_dataset_test_loader, device))
                        student_outputs['extrapolation'].append(test_regression(model, torch_dataset_extrapolation_loader, device))
                else: #elif args.setting == 'classification':
                    student_outputs['train'].append(test_classification(model, torch_dataset_train_loader, device))
                    student_outputs['test'].append(test_classification(model, torch_dataset_test_loader, device))
                    student_outputs['extrapolation'].append(test_classification(model, torch_dataset_extrapolation_loader, device))
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    for _ in range(args.epochs):
                        train_classification(model, optimizer, torch_dataset_train_loader, device)
                        student_outputs['train'].append(test_classification(model, torch_dataset_train_loader, device))
                        student_outputs['test'].append(test_classification(model, torch_dataset_test_loader, device))
                        student_outputs['extrapolation'].append(test_classification(model, torch_dataset_extrapolation_loader, device))
                if args.verbose:
                    print()
                    print('Student outputs:')
                    print('-' * 30)
                    print(student_outputs)
                # Save the student outputs and the trained model
                student_outputs_filename_prefix_ = f"{student_outputs_filename_prefix}/{int(time.time() * 1000)}"
                student_outputs_filename = f"{student_outputs_filename_prefix_}/student_outputs.pkl"
                student_outputs_filename_model = f"{student_outputs_filename_prefix_}/model.pt"
                writePickle(student_outputs, filename=student_outputs_filename)
                torch.save(model.state_dict(), student_outputs_filename_model)
    ###               
    return args.teacher_outputs_filename if args.save_file_destination else ''


if __name__ == '__main__':
    tmp_saved_filename = main()
    # sys.exit(predictions_filename)
    # Workaround to pass variable name to bash script...
    if tmp_saved_filename:
        with open('tmp_saved_filename.txt', 'w') as f:
            f.write(tmp_saved_filename)
