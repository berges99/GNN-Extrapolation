import time
import numpy as np
import scipy as sp

from tqdm import tqdm
from numba import jit



##########


def __computeNormalization(dist_matrix, i=10):
    '''Auxiliary function to find a typical distance in the dataset.
       To do so, we sort along the 0 axis and then average the distances we found at the i-th position.
       (Chunked version to avoid copies and problems with memory usage!)
    '''
    MAX_CHUNK_SIZE = 1_000
    m = dist_matrix.shape[1]
    usual_distances = np.zeros(m)
    for b in tqdm(range(0, m, MAX_CHUNK_SIZE)):
        usual_distances[b:min(b + MAX_CHUNK_SIZE, m - 1)] = \
            np.sort(dist_matrix[:, b:min(b + MAX_CHUNK_SIZE, m - 1)], axis=0)[i + 1, :]
    normalization_factor = np.mean(usual_distances)
    print(f'Using normalization factor at d = {normalization_factor}')
    return normalization_factor
    

def __computeThreshold(dist_matrix):
    '''Auxiliary function to find an adequate value to threshold the matrix such that
       we get approximately only 10% of non-zero values.
       (Chunked version to avoid copies and problems with memory usage!)
    '''
    MAX_CHUNK_SIZE = 1_000
    n, m  = dist_matrix.shape
    total = n * m
    counter = {}
    for b in tqdm(range(0, m, MAX_CHUNK_SIZE)):
        unique_b, counts_b = np.unique(
            dist_matrix[:, b:min(b + MAX_CHUNK_SIZE, m - 1)], return_counts=True)
        for k, v in zip(unique_b, counts_b):
            if k in counter: counter[k] += v
            else: counter[k] = v
    # Convert the dictionary into the previous (unique, counts) object
    unique, counts = [], []
    for element in sorted(counter.items()):
        unique.append(element[0])
        counts.append(element[1])
    count = 0
    for i in range(1, len(unique)): # We skip the first one (it will be 0 because of the diagonal)
        if i == len(unique) - 1:
            print('Thresholding is not possible with this distance matrix!')
            return np.inf, np.inf
        else:
            count += counts[i]
            if count / total >= 0.1:
                threshold = unique[i]
                print(f'Thresholding the distance matrix at d > {threshold} ({count / total:.2%} elements).')
                return threshold, count


@jit(nopython=True, fastmath=True)
def __computeW(dist_matrix, threshold, normalization):
    '''Auxiliary funtion to compute the adjacency matrix induced by the input distance matrix.'''
    n, m = dist_matrix.shape
    # If the matrix is squared, leverage symmetry
    for i in range(n):
        i_ = i if n == m else 0
        for j in range(i_, m):
            if threshold and dist_matrix[i, j] > threshold:
                dist_matrix[i, j] = 0
            elif normalization:
                dist_matrix[i, j] = np.exp(-(dist_matrix[i, j]**2 / (2 * normalization**2)))
            else:
                dist_matrix[i, j] = np.exp(-(dist_matrix[i, j]))
            if n == m:
                dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix


# TBD Adapt for huge matrices -> HDF5 form with pytables
def computeW(dist_matrix, threshold=True, normalization=True):
    '''
    Function that computes the induced adjacency matrix W by the pairwise relationships/distances
    between the data points in the dataset.

    Parameters:
        - dist_matrix: (np.ndarray) Input pairwise distances between data points in the dataset.
                                    Shape can be either (n x n) or (n x p) if Nystrom approximation is required.
        - threshold: (bool) Whether to threshold the distances.
        - normalization: (bool) Whether to normalize the exponentials.

    Returns:
        - (np.ndarray) Adjacency matrix induced by the pairwise distances.
                       Shape can be either (n x n) or (n x p) if Nystrom approximation is required.
    '''
    n, m = dist_matrix.shape
    # Compute the normalization factor
    if normalization:
        normalization = __computeNormalization(dist_matrix)
    # Compute threshold (so that approximately only 10% of the entries are non-zeros)
    if threshold:
        threshold, count = __computeThreshold(dist_matrix)
        # Check whether using sparse matrices is worth it
        if count * 3 <= n * m:
            print('Using sparse matrices...')
            use_sparse = True
        else:
            use_sparse = False
    # Apply the transformation to the dist matrix entry-wise
    dist_matrix = __computeW(dist_matrix, threshold=threshold, normalization=normalization)
    # Leverage sparsity if possible (otherwise returns dense matrix)
    return sp.sparse.csr_matrix(dist_matrix) if use_sparse else dist_matrix


##########


def computeD(W, C=None, output_format='sparse'):
    '''
    Auxiliary function to compute the diagonal matrix for a given adjacency matrix (weighted).

    Parameters:
        - W: (np.ndarray) Input weighted adjacency matris.
                          Shape can be either (n x n) or (p x p) if Nystrom approximation is required.
        - C: (np.ndarray) Nystrom sampled matrix.
                          Shape is (n x p)
        - output_format: (str) Whether to return the diagonal matrix in sparse or dense format.

    Returns:
        - (np.ndarray) or (sp.sparse.csr_matrix) Diagonal matrix of W.
    '''
    n = len(C) if C is not None else len(W)
    #diag = [np.sum(W[i, :]) for i in range(len(W))]
    if C is not None:
        diag = (C @ W @ (C.T @ np.ones((n, 1)))).reshape(-1)
    else:
        diag = (W @ np.ones((n, 1))).reshape(-1)
    if output_format == 'sparse':
        return sp.sparse.diags(diag, offsets=0, shape=(n, n), format='csr')
    elif output_format == 'dense': 
        return np.diag(diag)
    else:
        raise ValueError('Ouput format must be either "dense" or "sparse".')


##########


# TBD Adapt for huge matrices -> HDF5 form with pytables
def computeL(W, idxs=None, normalize=False):
    '''
    Function that computes the graph Laplacian, defined as L = D - W.

    Parameters:
        - W: (np.ndarray) Input adjacency matrix.
                          Shape can be either (n x n) or (n x p) if Nystrom approximation is required.
        - idxs: (np.array) Sampled columns for the Nystrom approximation.
        - normalize: (bool) Whether to use the normalized graph Laplacian.

    Returns:
        - (np.ndarray) or (sp.sparse.csr_matrix) 
          Diagonal matrix.
        - (tuple<np.ndarray>) or (tuple<sp.sparse.csr_matrix>) 
          Either the computed Laplacian or C & W_ (Nystrom)
    '''
    # if normalize:
    #     D_inv_sqrt = np.diag([D[i, i]**(-0.5) for i in range(len(L))])
    #     L = D_inv_sqrt @ L @ D_inv_sqrt
    n, m = W.shape
    # If full matrix (n x n) is given
    if n == m:
        D = computeD(W)
        return D, (D - W, None)
    # Nystrom approximation
    else:
        W_ = np.linalg.pinv(W[idxs, :]) # shape (p x p)
        D = computeD(W=W_, C=W)
        return D, (W, W_) # D, (C, W_)


##########
    

# TBD Adapt for huge matrices -> HDF5 form with pytables
def computeSmoothness(D, L, f):
    '''
    Smoothnesss with respect to the intrinsic structure of the data domain, which in our context is
    the weighted graph. We use the p-Dirichlet form of f with p = 2 (check https://arxiv.org/abs/1211.0053).

    Parameters:
        - D: (sp.sparse.csr_matrix) Sparse diagonal matrix of the graph.
        - L: (tuple<np.ndarray>) or (tuple<sp.sparse.csr_matrix)
             Graph combinatorial laplacian matrix (C, W_). If full Laplacian is given, C represents
             the full Laplacian and W_ is null. Otherwise, C is (n x p) and W_ is (p x p). 
        - f: (np.ndarray) Outputs/predictions/labels for the given data.
                          Shape: either (n x 1) for regression or (n x d) for classification.

    Returns:
        - (float) Notion of global "inverse" smoothness.
    '''
    # If Nystrom
    if L[1] is not None:
        return np.trace((f.T @ D @ f) - (f.T @ L[0] @ L[1] @ (L[0].T @ f)))
    else:
        return np.trace(f.T @ L[0] @ f)
