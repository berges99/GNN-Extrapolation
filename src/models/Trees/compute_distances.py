import time
import importlib
import numpy as np
import multiprocessing

from tqdm import tqdm
from joblib import Parallel, delayed



NUM_CORES = multiprocessing.cpu_count()


def auxiliaryDistParallel(trees, computeEditDistance):
    '''Auxiliary function that computes the pairwise distances for a single vector.'''
    n = len(trees)
    distances = np.zeros(n)
    for j in range(n):
        distances[j] = computeEditDistance(trees[0], trees[j])
    return distances


def computeDistMatrix(trees, method='apted', nystrom=False, parallel=False):
    '''
    Compute pairwise distances between all input trees.

    Parameters:
        - trees: (list<obj>) List with trees written in bracket format.
        - method: (str) Name of the m
        - nystrom: (bool) Whether to use the nyström approximation for computing the kernel.
        - parallel: (bool) Whether to parallelize the matrix computation in the CPU.

    Returns:
        - (np.ndarray) Matrix with the computed pairwise distanes.
    '''
    print()
    print('-' * 30)
    print('Computing the pairwise distance matrix between all rooted trees of the dataset...')
    n = len(trees)
    distances = np.zeros((n, n))
    # Import the method for compute the edit distance between a pair of trees
    computeEditDistance = getattr(
        importlib.import_module(f'models.Trees.edit_distance.{method}'), 'computeEditDistance')
    # Nyström approximation
    if nystrom:
        pass
    else:
        # If parallel computation is specified
        if parallel:
            distances_ = Parallel(n_jobs=NUM_CORES)(delayed(auxiliaryDistParallel)(trees[i:], computeEditDistance) for i in tqdm(range(n)))
            # Convert into symmetric matrix
            for i in range(n):
                distances[i, i:] = distances_[i]
            distances = np.where(distances, distances, distances.T)
        else:
            # Use the fact that it is a symmetric matrix
            for i in tqdm(range(n)):
                for j in range(i, n):
                    distances[i, j] = computeEditDistance(trees[i], trees[j])
                    distances[j, i] = distances[i, j]
    return distances
