import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from collections import OrderedDict

from utils.io import readPickle, writePickle, KeepOrderAction, booleanString
from utils.convert import getAdjacencyList, fromNetworkx2Torch, fromNetworkx2GraphML
from node_representations.compute_node_representations import computeDatasetNodeRepresentations



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    ##########
    # Parse all the input dataset related arguments
    parser.add_argument(
        '--dataset_filename', type=str, default='', 
        help='Complete relative path of the dataset to be used.')
    parser.add_argument(
        '--verbose', type=booleanString, default=False, 
        help='Whether to print the outputs through the terminal.')
    parser.add_argument(
        '--save_file_destination', type=booleanString, default=False,
        help='Whether to save the file path destination into a temporary file for later pipelined processing.')
    ##########
    # Parse all the arguments related to the embedding scheme, methods and distances
    subparsers = parser.add_subparsers(dest='embedding_scheme')
    subparsers.required = True
    ###
    WL = subparsers.add_parser('WL', help='WL kernel embedding_scheme parser.')
    WL.add_argument(
        '--method', type=str, required=True, choices=['continuous', 'hashing'], action=KeepOrderAction, #default='continuous',
        help='Method to compute the WL kernel. Available choices are [continuous, categorical].')
    WL.add_argument(
        '--depth', type=int, required=True, action=KeepOrderAction, #default=3,
        help='Max. receptive field depth for extracting the node representations, e.g. depth of the rooted trees.')
    WL.add_argument(
        '--initial_relabeling', type=str, required=True, choices=['ones', 'degrees'], action=KeepOrderAction, #default=ones
        help='Type of labeling to be used in the case that there aren\'t any available. Available choices are [ones, degrees].')
    WL.add_argument(
        '--normalization', type=str, required='--method continuous' in ' '.join(sys.argv),
        choices=['wasserstein', 'GCN'], action=KeepOrderAction, #default='wasserstein',
        help='Normalization to apply at each step of the WL kernel. Available choices are [wasserstein, GCN].')
    ###
    rooted_trees = subparsers.add_parser('Trees', help='Rooted trees embedding_scheme parser.')
    rooted_trees.add_argument(
        '--method', type=str, required=True, choices=['apted'], action=KeepOrderAction, #default='apted',
        help='Method to use for the extraction of rooted trees/d-patterns. Available choices are [apted].')
    rooted_trees.add_argument(
        '--depth', type=int, required=True, action=KeepOrderAction, #default=3,
        help='Max. receptive field depth for extracting the node representations, e.g. depth of the rooted trees.')
    return parser.parse_args()


def main():
    args = readArguments()
    networkx_dataset = readPickle(args.dataset_filename)
    # Flatten train + test + extrapolation data for computing the representations in one go
    networkx_dataset_flatten_idxs = [len(networkx_dataset[k]) for k in ['train', 'test']]
    networkx_dataset_flatten = [networkx_dataset[k] for k in ['train', 'test']]
    networkx_dataset_flatten = [G for k in networkx_dataset_flatten for G in k]
    # Append the extrapolation ones
    networkx_dataset_flatten_idxs_extrapolation = [
        len(networkx_dataset['extrapolation'][k]) for k in range(len(networkx_dataset['extrapolation']))]
    networkx_dataset_flatten_extrapolation = [G for k in networkx_dataset['extrapolation'] for G in k]
    networkx_dataset_flatten_idxs += networkx_dataset_flatten_idxs_extrapolation
    networkx_dataset_flatten += networkx_dataset_flatten_extrapolation
    # Fromat the data in a convenient way
    if args.embedding_scheme == 'WL':
        if args.method == 'continuous':
            formatted_dataset = fromNetworkx2Torch(networkx_dataset_flatten, initial_relabeling=args.initial_relabeling)
        else: # elif args.method == 'hashing'
            formatted_dataset = fromNetworkx2GraphML(networkx_dataset_flatten)
    else: # elif args.embedding_scheme == 'Trees'
        formatted_dataset = [getAdjacencyList(G) for G in networkx_dataset_flatten]
    ##########
    # Compute all the node representations for every node in the dataset # (if necessary)
    ordered_args = np.array([x[0] for x in args.ordered_args])
    node_representations_kwargs_idx = np.where(ordered_args == 'method')[0][0]
    node_representations_kwargs = OrderedDict({
        k: v for k, v in args.ordered_args[node_representations_kwargs_idx + 1:]
    })
    node_representations_filename = \
        f"{'/'.join(args.dataset_filename.split('/')[:-1])}/node_representations/{args.embedding_scheme}/{args.method}/" \
        f"{'_'.join([k[0] + str(v).capitalize() for k, v in node_representations_kwargs.items()])}/node_representations.pkl"
    node_representations = computeDatasetNodeRepresentations(
        formatted_dataset, args.embedding_scheme, args.method, 
        parallel=False, **node_representations_kwargs
    )
    # Reposition representations in {'train': [], 'test': [], 'extrapolation': [[], [], ...]}
    node_representations_extrapolation = []
    offset = networkx_dataset_flatten_idxs[0] + networkx_dataset_flatten_idxs[1]
    for k in networkx_dataset_flatten_idxs[2:]:
        node_representations_extrapolation.append(node_representations[offset:offset + k])
        offset += k
    node_representations = {
        'train': node_representations[:networkx_dataset_flatten_idxs[0]],
        'test': node_representations[
            networkx_dataset_flatten_idxs[0]:networkx_dataset_flatten_idxs[0] + networkx_dataset_flatten_idxs[1]],
        'extrapolation': node_representations_extrapolation
    }
    writePickle(node_representations, filename=node_representations_filename)
    if args.verbose:
        print()
        print('Node representations:')
        print('-' * 30)
        print(node_representations)
    return node_representations_filename if args.save_file_destination else ''


if __name__ == '__main__':
    tmp_saved_filename = main()
    # sys.exit(predictions_filename)
    # Workaround to pass variable name to bash script...
    if tmp_saved_filename:
        with open('tmp_saved_filename.txt', 'w') as f:
            f.write(tmp_saved_filename)
