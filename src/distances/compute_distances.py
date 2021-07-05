import time
import numpy as np
import scipy as sp
import tables as tb

from pathlib import Path



MAX_MEMORY = 10 # GB
MAX_DISK_STORAGE = 1_000 # GB


def resolveDistanceKwargs(distance, m, distance_kwargs):
    '''Auxiliary funtion to resolve the input keyword arguments to the main function, and 
       return interpretable kwargs for scipy.spatial.distance.cdist().'''
    cdist_kwargs = {}
    if distance == 'l2':
        cdist_kwargs['metric'] = 'euclidean'
    elif distance == 'l1':
        cdist_kwargs['metric'] = 'minkowski'
        cdist_kwargs['p'] = 1
    else:
        cdist_kwargs['metric'] = 'hamming'
        if 'scaling' in distance_kwargs:
            alphas = np.ones(m)
            for i in range(1, m):
                alphas[i] = alphas[i - 1] / distance_kwargs['scaling']
            cdist_kwargs['w'] = alphas
    return cdist_kwargs


# TBD Adapt test matrices to Nystrom scenario
def computeDistMatrix(node_representations,
                      distance,
                      save_destination,
                      test_idx=None,
                      nystrom=False,
                      float_precision=64,
                      output_format='npz',
                      **distance_kwargs):
    '''
    Compute pairwise distances between all input node representations.

    Parameters:
        - node_representations: (array_like) List with all node representations given in the correct format
                                             for the desired distance (flattened array).
        - distance: (str) Name of the distance to be used.
        - save_destination: (str) Path to where the resulting distance matrix is going to be stored.
        - test_idx: (int) Index denoting partition between train and test data.
        - nystrom: (bool) Whether to use the nyström approximation for computing the kernel.
        - output_format: (str)
        - float_precision: (int)
        (**distance_kwargs) Additional specific arguments for the chosen distance metric.

    Returns:
        -
    '''
    assert output_format in ['npz', 'h5'], 'Output format must be either "npy" or "h5"!'
    assert float_precision in [32, 64], 'Float precision must be either 32 or 64 bits!'
    dtype = np.float32 if float_precision == 32 else np.float64
    # Convert all representations into typed np.ndarrays
    node_representations = np.array([
        np.array(x_i, dtype=dtype) for x_i in node_representations], dtype=dtype)
    # num total representations & dimension of the representations
    n, m = node_representations.shape
    # If Nyström is specified
    if test_idx:
        matrix_size = (test_idx, n - test_idx)
        estimated_size = (float_precision / 8) * n * (n - test_idx) / 10**9 # GB
    elif nystrom:
        # Number of samples for the Nyström approximation (~ sqrt(n))
        p = int(np.ceil(np.sqrt(n)))
        matrix_size = (n, p)
        estimated_size = (float_precision / 8) * n * p / 10**9 # GB
    else:
        matrix_size = (n, n)
        estimated_size = (float_precision / 8) * n**2 / 10**9 # GB
    print()
    print(f'Total number of node representations: {n:,}')
    print(f'Estimated size of the {matrix_size} dense matrix using {dtype}: {estimated_size:,}GB')
    # Safety check
    assert estimated_size < MAX_DISK_STORAGE, 'Matrix size is too big! Consider augmenting MAX_STORAGE!'
    if estimated_size >= MAX_MEMORY and output_format != 'h5':
        raise ValueError(
            'Not enough in-memory size for this configuration! Consider using disk storage with hd5!')
    # Resolve the cdist parameters
    cdist_kwargs = resolveDistanceKwargs(distance, m, distance_kwargs)
    print()
    print('Computing all the indicated pairwise distances...')
    if test_idx:
        dist_matrix = sp.spatial.distance.cdist(
            node_representations[:test_idx, :], node_representations[test_idx:, :], **cdist_kwargs).astype(dtype)
    elif nystrom:
        # Saple p random columns (without replacement)
        idxs = np.arange(0, n, 1, dtype=int)
        np.random.shuffle(idxs)
        idxs = np.sort(idxs)[:p]
        dist_matrix = sp.spatial.distance.cdist(
            node_representations, node_representations[idxs, :], **cdist_kwargs).astype(dtype)
    else:
        dist_matrix = sp.spatial.distance.cdist(
            node_representations, node_representations, **cdist_kwargs).astype(dtype)
    # Chunked version for huge matrices
    if output_format == 'hd5':
        raise ValueError('Method with hd5 storage not implemented!')
    # Full in-memory approach
    else:
        # Save file in memory
        Path(save_destination).mkdir(parents=True, exist_ok=True)
        if nystrom:
            if test_idx:
                save_destination = f'{save_destination}/nystrom{float_precision}_{int(time.time() * 1000)}_test.npz'
            else:
                save_destination = f'{save_destination}/nystrom{float_precision}_{int(time.time() * 1000)}.npz'
            np.savez_compressed(save_destination, dist_matrix=dist_matrix, idxs=idxs)
        else:
            if test_idx:
                save_destination = f'{save_destination}/full{float_precision}_test.npz'
            else:
                save_destination = f'{save_destination}/full{float_precision}.npz'
            np.savez_compressed(save_destination, dist_matrix=dist_matrix)
        return dist_matrix
