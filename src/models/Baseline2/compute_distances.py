import time
import importlib
import numpy as np
import multiprocessing

from tqdm import tqdm
from joblib import Parallel, delayed



NUM_CORES = multiprocessing.cpu_count()


def auxiliaryDistParallel(node_representations, computeDistance, **distance_kwargs):
    '''Auxiliary function that computes the pairwise distances for a single vector.'''
    n = len(node_representations)
    distances = np.zeros(n)
    for j in range(n):
        distances[j] = computeDistance(
            node_representations[0], node_representations[j], **distance_kwargs)
    return distances


def computeDistMatrix(node_representations, 
                      embedding_scheme,
                      method, 
                      nystrom=False, 
                      parallel=False,
                      **distance_kwargs):
    '''
    Compute pairwise distances between all input node representations.

    Parameters:
        - node_representations: (array_like) List with all node representations given in the correct format
                                             for the desired distance (flattened array).
        - embedding_scheme: (str) Embedding scheme to be used, e.g. WL or Trees.
        - method: (str) Name of the distance to be used.
        - nystrom: (bool) Whether to use the nyström approximation for computing the kernel.
        - parallel: (bool) Whether to parallelize the matrix computation in the CPU.
        - (**distance_kwargs) Additional specific arguments for the chosen distance metric.

    Returns:
        - (np.ndarray) Matrix with the computed pairwise distanes.
    '''
    print()
    print('-' * 30)
    print('Computing the pairwise distance matrix between all node representations of the dataset...')
    n = len(node_representations)
    distances = np.zeros((n, n))
    # Import the method for compute the edit distance between a pair of node representations
    computeDistance = getattr(
        importlib.import_module(f'models.Baseline.{embedding_scheme}.dist.{method}'), 'computeDistance')
    # Nyström approximation
    if nystrom:
        pass
    else:
        # If parallel computation is specified
        if parallel:
            distances_ = \
                (Parallel(n_jobs=NUM_CORES)
                         (delayed(auxiliaryDistParallel)
                         (node_representations[i:], computeDistance, **distance_kwargs) for i in tqdm(range(n))))
            # Convert into symmetric matrix
            for i in range(n):
                distances[i, i:] = distances_[i]
            distances = np.where(distances, distances, distances.T)
        else:
            # Use the fact that it is a symmetric matrix
            for i in tqdm(range(n)):
                for j in range(i, n):
                    distances[i, j] = computeDistance(
                        node_representations[i], node_representations[j], **distance_kwargs)
                    distances[j, i] = distances[i, j]
    return distances
