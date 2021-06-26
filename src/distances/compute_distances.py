import time
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
    distances = np.zeros(n)
    for j in range(n):
        distances[j] = computeDistance(
            node_representations[0], node_representations[j], **distance_kwargs)
    return distances


# TBD Adapt to futer distance argument implementations (**distance_kwargs)
@jit(nopython=True, parallel=True)
def auxiliaryDistFullNumba(node_representations, computeDistanceNumba, scaling=0):
    ''''''
    n = len(node_representations)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if scaling > 0:
                distances[i, j] = computeDistanceNumba(
                    node_representations[i], node_representations[j], scaling)
            else:
                distances[i, j] = computeDistanceNumba(
                    node_representations[i], node_representations[j])
            distances[j, i] = distances[i, j]
        if i % 100 == 0: print(i)
    return distances



def computeDistMatrix(node_representations,
                      distance,
                      nystrom=False, 
                      parallel=False,
                      numba=False,
                      **distance_kwargs):
    '''
    Compute pairwise distances between all input node representations.

    Parameters:
        - node_representations: (array_like) List with all node representations given in the correct format
                                             for the desired distance (flattened array).
        - distance: (str) Name of the distance to be used.
        - nystrom: (bool) Whether to use the nyström approximation for computing the kernel.
        - parallel: (bool) Whether to parallelize the matrix computation in the CPU.
        - (**distance_kwargs) Additional specific arguments for the chosen distance metric.

    Returns:
        - (np.ndarray) Matrix with the computed pairwise distanes.
    '''
    print()
    print('-' * 30)
    print('Computing the pairwise distance matrix between all node representations of the dataset...')
    # Nyström approximation
    if nystrom:
        pass
    # Full pairwise matrix computation (exact)
    else:
        # Numba c++ vectorization
        if numba:
            # Convert all representations into typed np.arrays
            node_representations = np.array([np.array(x_i, dtype=float) for x_i in node_representations], dtype=float)
            # Import the numba-rized method
            computeDistanceNumba = getattr(importlib.import_module(f'distances.methods'), f'{distance}Numba')
            # Resolve extra arguments (numba/c++ do not accept python objects as arguments...)
            scaling = 0 if distance != 'hamming' else distance_kwargs['scaling']
            distances = auxiliaryDistFullNumba(node_representations, computeDistanceNumba, scaling=scaling)
        # Python approach
        else:
            n = len(node_representations)
            distances = np.zeros((n, n))
            computeDistance = getattr(importlib.import_module(f'distances.methods'), distance)
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
