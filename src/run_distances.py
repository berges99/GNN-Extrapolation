import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from collections import OrderedDict

from utils.stats import *
from utils.io import KeepOrderAction, readPickle, writePickle

from distances.compute_distances import computeDistMatrix


#from models.Baseline.compute_distances import computeDistMatrix




def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    ##########
    parser.add_argument(
        '--node_representations_filename', type=str, required=True,
        help='Full relative path to the node representations.')
    parser.add_argument(
        '--verbose', '-v', type=bool, default=True, 
        help='Whether to print the outputs through the terminal.')
    ##########
    # Parse all the arguments related to the distance computation
    parser.add_argument(
        '--distance', type=str, required=True, choices=['hamming', 'l1', 'l2', 'edit'], action=KeepOrderAction,
        help='Distance to use for computing the distances between node representations. Available choices are [hamming, l1, l2].')
    parser.add_argument(
        '--scaling', type=str, required='--distance hamming' in ' '.join(sys.argv),
        choices=['constant', 'maxdegree', 'avgdegree'], action=KeepOrderAction,
        help='Apply special scaling to the distance between node representations. Available choices are [max_degree].')
    parser.add_argument(
        '--relabel', type=bool, required='--distance edit' in ' '.join(sys.argv), action=KeepOrderAction,
        help='Whether to perform relabeling of the extracted rooted trees of the dataset, i.e. no relabel cost in edit distance.')
    return parser.parse_args()


def main():
    args = readArguments()
    # Read actual dataset (for getting basic stats like max_degree, avg_degree, ...)
    dataset_filename = \
        f"{'/'.join(args.node_representations_filename.split('/')[:-4])}" \
        f"/raw/{args.node_representations_filename.split('__')[-1]}"
    networkx_dataset = readPickle(dataset_filename)
    ##########
    # Read the node representations
    node_representations = readPickle(args.node_representations_filename)
    # Compute the pairwise distance matrix if necessary
    ordered_args = np.array([x[0] for x in args.ordered_args])
    distance_kwargs_idx = np.where(ordered_args == 'distance')[0][0]
    distance_kwargs = OrderedDict({
        k: v for k, v in args.ordered_args[distance_kwargs_idx + 1:]
    })
    distances_filename = \
        f"{'/'.join(args.node_representations_filename.split('/')[:-1])}" \
        f"/dist_matrices/{args.distance}/" \
        f"{args.node_representations_filename.split('__')[0].split('/')[-1]}" \
        f"__{'_'.join([k[0] + str(v).capitalize() for k, v in distance_kwargs.items()])}" \
        f"__{args.node_representations_filename.split('__')[1]}"
    if os.path.isfile(distances_filename):
        node_representations_flatten, dist_matrix = readPickle(distances_filename)
    else:
        node_representations_flatten = [item for sublist in node_representations for item in sublist]
        if 'scaling' in distance_kwargs:
            if distance_kwargs['scaling'] == 'maxdegree':
                distance_kwargs['scaling'] = getMaxDegree(networkx_dataset)
            elif distance_kwargs['scaling'] == 'avgdegree':
                distance_kwargs['scaling'] = getAvgDegree(networkx_dataset)
            else:
                distance_kwargs['scaling'] = 1
        dist_matrix = computeDistMatrix(
            node_representations_flatten, args.distance, nystrom=False, parallel=True, **distance_kwargs)
        writePickle((node_representations_flatten, dist_matrix), filename=distances_filename)
    if args.verbose:
        print()
        print(f'Distance matrix: (shape = [{dist_matrix.shape}])')
        print('-' * 30)
        print(dist_matrix)


if __name__ == '__main__':
    main()
