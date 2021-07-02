import numpy as np
import scipy as sp

from numba import jit



def computeD(W, output_format='sparse'):
    '''Auxiliary function to compute the diagonal matrix for a given adjacency matrix (weighted).'''
    n = len(W)
    #diag = [np.sum(W[i, :]) for i in range(len(W))]
    diag = (W @ np.ones((n, 1))).reshape(-1)
    if output_format == 'sparse':
        return sp.sparse.diags(diag, offsets=0, shape=(n, n), format='csr')
    elif output_format == 'dense': 
        return np.diag(diag)
    else:
        raise ValueError('Ouput format must be either "dense" or "sparse".')
    

# TBD Adapt for huge matrices -> HDF5 form with pytables
# TBD Use the thresholded case -> maybe it would increase substantially the performance using scipy csr matrices
@jit(nopython=True, parallel=True)
def computeWNumba(dist_matrix, threshold=0, normalization_factor=0):
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
    n = len(dist_matrix)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if threshold != 0 and threshold > dist_matrix[i, j]:
                W[i, j] = 0
            else:
                if normalization_factor != 0:
                    W[i, j] = np.exp(-(dist_matrix[i, j]**2 / normalization_factor))
                else:
                    W[i, j] = np.exp(-dist_matrix[i, j])
            W[j, i] = W[i, j]
    return W


# TBD Adapt for huge matrices -> HDF5 form with pytables
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


# TBD Adapt for huge matrices -> HDF5 form with pytables
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