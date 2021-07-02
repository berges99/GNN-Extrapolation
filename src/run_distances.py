import os
import sys
import argparse
import numpy as np

from collections import OrderedDict

from utils.stats import getMaxDegree, getAvgDegree
from utils.io import KeepOrderAction, readPickle, booleanString

from distances.compute_distances import computeDistMatrix



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    ##########
    parser.add_argument(
        '--node_representations_filename', type=str, required=True,
        help='Full relative path to the node representations.')
    parser.add_argument(
        '--verbose', '-v', type=booleanString, default=False, 
        help='Whether to print the outputs through the terminal.')
    parser.add_argument(
        '--save_file_destination', type=booleanString, default=False,
        help='Whether to save the file path destination into a temporary file for later pipelined processing.')
    ##########
    parser.add_argument(
        '--nystrom', type=booleanString, default=False, 
        help='Whether to use Nystrom approximation or not.')
    parser.add_argument(
        '--float_precision', type=int, choices=[32, 64], default=64, 
        help='Precision of the float numbers in the distance matrix.')
    parser.add_argument(
        '--output_format', type=str, choices=['npz', 'hd5'], default='npz', 
        help='Desired format for the output file stored in memory.')
    ##########
    # Parse all the arguments related to the distance computation
    parser.add_argument(
        '--distance', type=str, required=True, choices=['hamming', 'l1', 'l2', 'edit'], action=KeepOrderAction,
        help='Distance to use for computing the distances between node representations.')
    parser.add_argument(
        '--scaling', type=str, required='--distance hamming' in ' '.join(sys.argv),
        choices=['constant', 'maxdegree', 'avgdegree'], action=KeepOrderAction,
        help='Apply special scaling to the distance between node representations.')
    parser.add_argument(
        '--relabel', type=booleanString, required='--distance edit' in ' '.join(sys.argv), action=KeepOrderAction,
        help='Whether to perform relabeling of the extracted rooted trees of the dataset, i.e. no relabel cost in edit distance.')
    return parser.parse_args()


def main():
    args = readArguments()
    # Read actual dataset (for getting basic stats like max_degree, avg_degree, ...)
    networkx_dataset = readPickle(f"{'/'.join(args.node_representations_filename.split('/')[:-5])}/raw_networkx.pkl")
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
        f"{'/'.join(args.node_representations_filename.split('/')[:-1])}/dist_matrices/{args.distance}"
    if distance_kwargs:
        distances_filename = \
            f"{distances_filename}/{'_'.join([k[0] + str(v).capitalize() for k, v in distance_kwargs.items()])}"
    node_representations_flatten = [item for sublist in node_representations for item in sublist]
    # Determine scaling (dataset dependent)
    if 'scaling' in distance_kwargs:
        if distance_kwargs['scaling'] == 'maxdegree':
            distance_kwargs['scaling'] = getMaxDegree(networkx_dataset)
        elif distance_kwargs['scaling'] == 'avgdegree':
            distance_kwargs['scaling'] = getAvgDegree(networkx_dataset)
        else:
            distance_kwargs['scaling'] = 1
    dist_matrix = computeDistMatrix(
        node_representations_flatten, distance=args.distance, save_destination=distances_filename,
        nystrom=args.nystrom, float_precision=args.float_precision, output_format=args.output_format, **distance_kwargs
    )
    if args.verbose:
        print()
        print(f'Distance matrix: (shape = [{dist_matrix.shape}])')
        print('-' * 30)
        print(dist_matrix)
    return distances_filename if args.save_file_destination else ''


if __name__ == '__main__':
    tmp_saved_filename = main()
    # sys.exit(predictions_filename)
    # Workaround to pass variable name to bash script...
    if tmp_saved_filename:
        with open('tmp_saved_filename.txt', 'w') as f:
            f.write(tmp_saved_filename)
