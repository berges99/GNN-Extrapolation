import time
import torch
import scipy
import importlib
import numpy as np
import multiprocessing

from tqdm import tqdm
from numba import jit
from joblib import Parallel, delayed



NUM_CORES = multiprocessing.cpu_count()


def auxiliaryDistParallel(node_representations, computeDistance, **distance_kwargs):
    '''Auxiliary function that computes the pairwise distances for a single vector.'''
    n = len(node_representations)
    distances = np.zeros(n - 1)
    for j in range(1, n):
        distances[j] = computeDistance(
            node_representations[0], node_representations[j], **distance_kwargs)
    return distances


def auxiliaryDistFull(node_representations, computeDistance, **distance_kwargs):
    '''Auxiliary function that computes all the pairswise distances in the given input list.'''
    n = len(node_representations)
    distances = np.zeros(int(0.5 * n * (n - 1)))
    idx = 0
    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            distances[idx] = computeDistance(
                node_representations[i], node_representations[j], **distance_kwargs)
            idx += 1
    return distances


# TBD Adapt to futer distance argument implementations (**distance_kwargs)
# Adapted numba-rized methods to accept extra parameter *alphas* !!!
@jit(nopython=True, parallel=True)
def auxiliaryDistFullNumba(node_representations, computeDistanceNumba, scaling=0):
    ''''''
    n = len(node_representations)
    distances = np.zeros(int(0.5 * n * (n - 1)))
    # Precompute weights
    alphas = np.ones(len(node_representations[0]))
    if scaling > 0:
        for i in range(1, len(alphas)):
            alphas[i] = alphas[i - 1] / scaling
    # Compute all the distance matrix
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            distances[idx] = computeDistanceNumba(
                node_representations[i], node_representations[j], alphas)
            idx += 1
        if i % 100 == 0: print(i)
    return distances


def computeDistMatrix(node_representations,
                      distance,
                      nystrom=False,
                      pytorch=True,
                      numba=False,
                      parallel=False,
                      **distance_kwargs):
    '''
    Compute pairwise distances between all input node representations.

    Parameters:
        - node_representations: (array_like) List with all node representations given in the correct format
                                             for the desired distance (flattened array).
        - distance: (str) Name of the distance to be used.
        - nystrom: (bool) Whether to use the nyström approximation for computing the kernel.
        - pytorch: (bool) Whether to use existing scipy/torch implementations for the "simple" distances.
        - numba: (bool) Whether to vectorize the python code into C++ with numba package.
        - parallel: (bool) Whether to parallelize the matrix computation in the CPU.
        - (**distance_kwargs) Additional specific arguments for the chosen distance metric.

    Returns:
        - (np.array) 1d-array with length 1/2 * n * (n - 1) representing the condensed pairwise distance matrix.
    '''
    print()
    print('-' * 30)
    print('Computing the pairwise distance matrix between all node representations of the dataset...')
    # Nyström approximation
    if nystrom:
        pass
    # Full pairwise matrix computation (exact)
    else:
        # Compute all the pairwise distances using torch/scipy functions
        if pytorch:
            # Convert all representations into typed np.arrays
            node_representations = np.array([np.array(x_i, dtype=float) for x_i in node_representations], dtype=float)
            # Convert into torch tensors
            # node_representations = torch.from_numpy(node_representations)
            # distances = torch.nn.functional.pdist(node_representations, p=1)
            # Resolve arguments
            pdist_kwargs = {}
            if distance == 'l2':
                pdist_kwargs['metric'] = 'euclidean'
            elif distance == 'l1':
                pdist_kwargs['metric'] = 'minkowski'
                pdist_kwargs['p'] = 1
            else:
                pdist_kwargs['metric'] = 'hamming'
                if 'scaling' in distance_kwargs:
                    alphas = np.ones(len(node_representations[0]))
                    for i in range(1, len(alphas)):
                        alphas[i] = alphas[i - 1] / distance_kwargs['scaling']
                    pdist_kwargs['w'] = alphas.copy()
            distances = scipy.spatial.distance.pdist(node_representations, **pdist_kwargs)
            #distances = scipy.spatial.distance.squareform(distances)
        # Numba c++ vectorization
        elif numba:
            # Convert all representations into typed np.arrays
            node_representations = np.array([np.array(x_i, dtype=float) for x_i in node_representations], dtype=float)
            # Import the numba-rized method
            computeDistanceNumba = getattr(importlib.import_module(f'distances.methods'), f'{distance}Numba')
            # Resolve extra arguments (numba/c++ do not accept python objects as arguments...)
            scaling = 0 if distance != 'hamming' else distance_kwargs['scaling']
            distances = auxiliaryDistFullNumba(node_representations, computeDistanceNumba, scaling=scaling)
        # Pythonic parallelized approach
        elif parallel:
            computeDistance = getattr(importlib.import_module(f'distances.methods'), distance)
            distances = \
                (Parallel(n_jobs=NUM_CORES)
                         (delayed(auxiliaryDistParallel)
                         (node_representations[i:], computeDistance, **distance_kwargs) for i in tqdm(range(n))))
            # Flatten distances
            distances = np.array([item for sublist in distances for item in sublist])
            # # Convert into symmetric matrix
            # for i in range(n):
            #     distances[i, i:] = distances_[i]
            # distances = np.where(distances, distances, distances.T)
        # Fully unoptimized pythonic way
        else:
            computeDistance = getattr(importlib.import_module(f'distances.methods'), distance)
            distances = auxiliaryDistFull(node_representations, computeDistance, **distance_kwargs)  
    return distances
