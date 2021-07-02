import numpy as np

from tqdm import tqdm
from numba import jit

from utils.smoothness import computeWNumba, computeLaplacian



def __fromNystrom2Full(C, idxs):
    '''Auxiliary function to go from Nystrom decomposition to the full matrix form.'''
    return C @ np.linalg.pinv(C[idxs, :]) @ C.T


# TBD Adapt for huge matrices -> HD5 form with pytables
@jit(nopython=True, parallel=True)
def fullBaseline(dist_matrix, teacher_outputs_flatten, method='knn'):
    '''Auxiliary function that computes the base/knn baseline using the predictions of the closest node representations.'''
    n = len(teacher_outputs_flatten)
    student_outputs = np.zeros(n)
    # Set diagonal to infinity, such that we cannot use the real outputs of the values we want to predict
    np.fill_diagonal(dist_matrix, np.inf)
    for i in range(n):
        min_dist = np.min(dist_matrix[:, i])
        # Check if we want exact or fuzzy matching
        if method == 'knn' or min_dist == 0:
            min_idxs = np.where(dist_matrix[:, i] == min_dist)[0]
            prediction = np.mean(teacher_outputs_flatten[min_idxs])
            student_outputs[i] = prediction
        else:
            student_outputs[i] = np.mean(teacher_outputs_flatten)
    return student_outputs


# TBD Adapt for huge matrices -> HD5 form with pytables
@jit(nopython=True, parallel=True)
def nystromBaseline(C, idxs, teacher_outputs_flatten, method='knn'):
    '''Auxiliary function that computes the base/knn baseline using the predictions of the closest node representations,
       using a chunked version with the Nystrom algorithm.
    '''
    n = len(teacher_outputs_flatten)
    MAX_CHUNK_SIZE = 1_000
    C_W = C @ np.linalg.pinv(C[idxs, :]) # shape (n, p)
    student_outputs = np.zeros(n)
    for b in range(0, n, MAX_CHUNK_SIZE):
        dist_matrix_b = C_W @ C.T[:, b:b + MAX_CHUNK_SIZE] # shape (n, MAX_CHUNK_SIZE)
        b //= MAX_CHUNK_SIZE
        # Set diagonal to infinity, such that we cannot use the real outputs of the values we want to predict
        np.fill_diagonal(dist_matrix_b[b * MAX_CHUNK_SIZE:], np.inf)
        for i in range(MAX_CHUNK_SIZE):
            min_dist = np.min(dist_matrix_b[:, i])
            # Check if we want exact or fuzzy matching
            if method == 'knn' or min_dist == 0:
                min_idxs = np.where(dist_matrix_b[:, i] == min_dist)[0]
                prediction = np.mean(teacher_outputs_flatten[min_idxs])
                student_outputs[b * MAX_CHUNK_SIZE + i] = prediction
            else:
                student_outputs[b * MAX_CHUNK_SIZE + i] = np.mean(teacher_outputs_flatten)
    return student_outputs


# # TBD Adapt for huge matrices -> HDF5 form with pytables
# def smoothBaseline(dist_matrix, student_outputs, test_node_idxs, smoothing, **smoothing_kwargs):
#     '''
#     Smooth the baseline outputs with some transformation, such that f = h(L)y.
    
#     Parameters:
#         - dist_matrix: (np.ndarray) Full pairwise distance matrix between input node representations.
#         - student_outputs: (np.array) Array with the predictions of the basic baseline.
#         - test_node_idxs
#         - smoothing: (str) Type of smoothing to aply;
#                            · tikhonov -> h(L) := 1 / (1 + gamma * L) Can be viewed as a low pass filter.
#                            · heat_kernel -> h(L) := exp(tau * L) Can be viewed as a heat diffusion operator.
#                            · approx_kernel
#         (**smoothing_kwargs) Additional keyword arguments for the smoothing transform.

#     Returns:
#         - (np.array) Smoothed student outputs.
#     '''
#     # Compute the normalized graph combinatorial laplacian induced by the given dist matrix
#     W = computeWNumba(dist_matrix)
#     print('Computed W!')
#     L = computeLaplacian(W, normalize=False)
#     print('Computed laplacian!')
#     # Tikhonov regularization
#     if smoothing == 'tikhonov':
#         L = 1 + smoothing_kwargs['gamma'] * L
#         print(L)
#         h = np.linalg.inv(L)
#         print(h)
#         f = h @ student_outputs
#         print(f)
#     return student_outputs


# TBD Adapt for huge matrices -> HDF5 form with pytables
def Baseline(teacher_outputs_flatten, 
             dist_matrix,
             idxs=None,
             num_outputs=1, # This parameter implicitly indicates the setting
             method='knn',
             smoothing=None,
             **smoothing_kwargs):
    '''
    Implementation of the baseline model. It gives student_outputs for the test data based on node representation
    similarities.

    Parameters:
        - teacher_outputs_flatten: (np.array<float>) Array with all the flattened outputs for all the nodes in the dataset.
        - dist_matrix: (np.ndarray) Pairwise distance matrix between all nodes in the dataset.
        - idxs: (np.array) When using nystrom, the indices of the sampled columns.
        - method: (str) Whether to use the baseline baseline or baseline knn for producing the initial outputs.
        - smoothing: (str) Which type of smoothing to apply (if necessary).
        (**smoothing_kwargs) Additional keyword arguments for the smoothing transform.

    Returns:
        - (np.array) Flattened student_outputs for the test data.
    '''
    # If nystrom
    if idxs is not None:
        student_outputs = nystromBaseline(dist_matrix, idxs, teacher_outputs_flatten, method=method)
    # Otherwise the full matrix is given
    else:
        student_outputs = fullBaseline(dist_matrix, teacher_outputs_flatten, method=method)
    # ##########
    # # Smooth the baseline with some kernels if necessary
    # if smoothing:
    #     student_outputs = smoothBaseline(
    #         dist_matrix, student_outputs, test_node_idxs, smoothing=smoothing, **smoothing_kwargs)
    return student_outputs
