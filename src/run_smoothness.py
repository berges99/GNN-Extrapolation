import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

from utils.io import readPickle, writePickle



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dist_matrix_filename', type=str, required=True,
        help='Full relative path to the node representations to be used by the baseline method.')
    parser.add_argument(
        '--teacher_outputs_filename', type=str, required=True, 
        help='Full relative path to the teacher outputs of the given dataset (or parent directory path).')
    parser.add_argument(
        '--model', type=str, required=True, choices=['Baseline', 'GIN'],
        help='Model used for the predictions.')
    ###
    parser.add_argument(
        '--verbose', '-v', type=bool, default=True, 
        help='Whether to print the outputs through the terminal.')
    return parser.parse_args()


##########


def computeD(W):
    '''Auxiliary function to compute the diagonal matrix for a given adjacency matrix (weighted).'''
    return np.diag([np.sum(W[i, :]) for i in range(len(W))])


def auxiliaryGaussianKernel(dist, threshold=None, normalization_factor=None):
    '''Auxiliary function that computes the thresholded Gaussian kernel weighting function.'''
    if threshold and threshold > dist:
        return 0
    else:
        if normalization_factor:
            return np.exp(-(dist**2 / normalization_factor))
        else:
            return np.exp(-dist)


def computeW(dist_matrix, threshold=None, normalization_factor=None):
    '''
    Function that computes the induced adjacency matrix W by the pairwise relationships/distances
    between the data points in the dataset.

    Parameters:
        - dist_matrix: (np.ndarray) Input pairwise distances between data points in the dataset.
        - threshold: Threshold for the distances.
        - normalization_factor: Normalization factor for the weighting function.

    Returns:
        - (np.ndarray) Adjacency matrix induced by the pairwise distances.
    '''
    W = np.zeros_like(dist_matrix)
    for i in range(len(W)):
        for j in range(len(W)):
            W[i, j] = auxiliaryGaussianKernel(
                dist_matrix[i, j], threshold=threshold, normalization_factor=normalization_factor)
            W[j, i] = W[i, j]
    return W


def computeLaplacian(W, normalize=False):
    '''
    Function that computes the graph Laplacian, defined as L = D - W.

    Parameters:
        - W: (np.ndarray) Input adjacency matrix.
        - normalize: (bool) Whether to use the normalized graph Laplacian.

    Returns:
        - (np.ndarray) Graph Laplacian of the given adjacency matrix.
    '''
    D = computeD(W)
    L = D - W
    if normalize:
        D_inv_sqrt = np.diag([D[i, i]**(-0.5) for i in range(len(L))])
        L = D_inv_sqrt @ L @ D_inv_sqrt
    return L


def computeSmoothness(L, F):
    '''
    Smoothnesss with respect to the intrinsic structure of the data domain, which in our context is
    the weighted graph. We use the p-Dirichlet form of f with p = 2 (check https://arxiv.org/abs/1211.0053).

    Parameters:
        - L: (np.ndarray) Graph Laplacian matrix.
        - F: (np.ndarray) Outputs/predictions/labels for the given data.
                          Shape: either (n x 1) for regression or (n x d) for classification.

    Returns:
        - (float) Notion of global smoothness.
    '''
    return np.trace(F.T @ L @ F)


##########


def resolveTeacherOutputsFilenames(path):
    '''Auxiliary function that returns an iterable with all the involved teacher outputs.'''
    if os.path.isdir(path):
        teacher_outputs_filenames = [
            f"{path}/{t_i}/teacher_outputs.pkl" for t_i in os.listdir(path)
            if all(c.isdigit() for c in t_i[-13:])
        ]
    else:
        teacher_outputs_filenames = [path]
    return teacher_outputs_filenames


def main():
    args = readArguments()
    # Read the dist matrix and node representations for the specified data
    node_representations, dist_matrix = readPickle(args.dist_matrix_filename)
    # Compute the graph laplacian
    W = computeW(dist_matrix)
    L = computeLaplacian(W, normalize=True)
    # Read the teacher outputs of the dataset (iterate through folder if necessary)
    for teacher_outputs_filename in tqdm(resolveTeacherOutputsFilenames(args.teacher_outputs_filename)):
        # Read the teacher outputs
        teacher_outputs = readPickle(teacher_outputs_filename)
        # Student outputs to process
        student_outputs_filenames = [
            f"{'/'.join(teacher_outputs_filename.split('/')[:-1])}/student_outputs/{args.model}/{x}" 
            for x in os.listdir(f"{'/'.join(teacher_outputs_filename.split('/')[:-1])}/student_outputs/{args.model}")
            if x.endswith('.pkl')
        ]
        student_smoothness = defaultdict(lambda: defaultdict(list))
        for student_outputs_filename in student_outputs_filenames:
            student_outputs = readPickle(student_outputs_filename)
            # Resolve method characteristics
            k = student_outputs_filename.split('/')[-1].split('__epochs')[0].rstrip('.pkl')
            # Iterate through all the student outputs (if epochs > 1)
            for i, student_output in enumerate(student_outputs):
                if len(np.array(student_output).shape) == 2:
                    student_output = [item for sublist in student_output for item in sublist]
                student_output = np.array(student_output)[:, None]
                smoothness = computeSmoothness(L, F=student_output)
                student_smoothness[k][i].append(smoothness)
        # Compute mean and std deviation for all initializations
        for k, v in student_smoothness.items():
            for epoch, smoothness in v.items():
                student_smoothness[k][epoch] = {
                    'mean_smoothness': np.mean(smoothness),
                    'std_dev_smoothness': np.std(smoothness),
                }
            # Prepare data for pandas multiindex columns
            student_smoothness[k] = {
                (outer_key, inner_key): values 
                for outer_key, inner_dict in student_smoothness[k].items() 
                for inner_key, values in inner_dict.items()
            }
        # Convert into pandas dataframe
        student_smoothness = pd.DataFrame.from_dict(student_smoothness, orient='index')
        # Store file
        student_smoothness_filename = \
            f"{'/'.join(student_outputs_filenames[0].split('/')[:-1])}/smoothness_stats.csv"
        student_smoothness.to_csv(student_smoothness_filename)
                

if __name__ == '__main__':
    main()