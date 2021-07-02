import numpy as np

from utils.smoothness import computeWNumba, computeLaplacian



# TBD Adapt for huge matrices -> HDF5 form with pytables
def baseBaseline(dist_matrix, teacher_outputs_flatten, test_node_idxs, method='baseline'):
    '''Auxiliary function that computes the base/knn baseline using the predictions of the closest node representations.'''
    student_outputs = []
    for test_node_idx in test_node_idxs:
        min_dist = np.min(dist_matrix[:, test_node_idx])
        # Check if we want exact or fuzzy matching
        if method == 'knn' or min_dist == 0:
            min_idxs = np.where(dist_matrix[:, test_node_idx] == min_dist)[0]
            prediction = np.mean(teacher_outputs_flatten[min_idxs])
            student_outputs.append(prediction)
        else:
            student_outputs.append(np.mean(teacher_outputs_flatten))
    return student_outputs


# TBD Adapt for huge matrices -> HDF5 form with pytables
def smoothBaseline(dist_matrix, student_outputs, test_node_idxs, smoothing, **smoothing_kwargs):
    '''
    Smooth the baseline outputs with some transformation, such that f = h(L)y.
    
    Parameters:
        - dist_matrix: (np.ndarray) Full pairwise distance matrix between input node representations.
        - student_outputs: (np.array) Array with the predictions of the basic baseline.
        - test_node_idxs
        - smoothing: (str) Type of smoothing to aply;
                           · tikhonov -> h(L) := 1 / (1 + gamma * L) Can be viewed as a low pass filter.
                           · heat_kernel -> h(L) := exp(tau * L) Can be viewed as a heat diffusion operator.
                           · approx_kernel
        (**smoothing_kwargs) Additional keyword arguments for the smoothing transform.

    Returns:
        - (np.array) Smoothed student outputs.
    '''
    # Compute the normalized graph combinatorial laplacian induced by the given dist matrix
    W = computeWNumba(dist_matrix)
    print('Computed W!')
    L = computeLaplacian(W, normalize=False)
    print('Computed laplacian!')
    # Tikhonov regularization
    if smoothing == 'tikhonov':
        L = 1 + smoothing_kwargs['gamma'] * L
        print(L)
        h = np.linalg.inv(L)
        print(h)
        f = h @ student_outputs
        print(f)
    return student_outputs


# TBD Adapt for huge matrices -> HDF5 form with pytables
def Baseline(node_representations_flatten,
             node_representations_idxs,
             teacher_outputs_flatten, 
             dist_matrix, 
             train_idxs, 
             test_idxs=None,
             num_outputs=1, # This parameter implicitly indicates the setting
             method='knn',
             smoothing=None,
             **smoothing_kwargs):
    '''
    Implementation of the baseline model. It gives student_outputs for the test data based on node representation
    similarities.

    Parameters:
        - node_representations_flatten: (array_like) List with all the flattened node representations in the dataset.
        - node_representations_idxs: (np.array<int>) Array with the number of nodes per graph in the dataset.
        - teacher_outputs_flatten: (np.array<float>) Array with all the flattened outputs for all the nodes in the dataset.
        - dist_matrix: (np.ndarray) Pairwise distance matrix between all nodes in the dataset.
        - train_idxs: (np.array) Graphs to be used as train data.
        - test_idxs: (np.array) Graphs to be used as test data.
        - method: (str) Whether to use the baseline baseline or baseline knn for producing the initial outputs.
        - smoothing: (str) Which type of smoothing to apply (if necessary).
        (**smoothing_kwargs) Additional keyword arguments for the smoothing transform.

    Returns:
        - (np.array) Flattened indices of the predicted values (within the main dataset).
        - (np.array) Flattened student_outputs for the test data.
    '''
    # Flatten the idxs
    train_node_idxs = []
    for train_graph_idx in train_idxs:
        n = node_representations_idxs[train_graph_idx]
        train_node_idxs.extend([x for x in range(train_graph_idx * n, train_graph_idx * n + n)])
    # Set diagonal values to infinity, such that we cannot use real output values for the student_outputs
    # Set test rows to infinity, such that we cannot use them for the student_outputs (if necessary)
    if test_idxs:
        test_node_idxs = []
        for test_graph_idx in test_idxs:
            n = node_representations_idxs[test_graph_idx]
            test_node_idxs.extend([x for x in range(test_graph_idx * n, test_graph_idx * n + n)])
        for test_node_idx in test_node_idxs:
            dist_matrix[test_node_idx, :] = [np.inf] * len(dist_matrix)
    else:
        test_node_idxs = train_node_idxs
    # Set diagonal to infinity, such that we cannot use the real outputs of the values we want to predict
    np.fill_diagonal(dist_matrix, np.inf)
    ##########
    # Produce the initial outputs (baseline baseline/knn)
    student_outputs = baseBaseline(
        dist_matrix, teacher_outputs_flatten, test_node_idxs, method=method)
    ##########
    # Smooth the baseline with some kernels if necessary
    if smoothing:
        student_outputs = smoothBaseline(
            dist_matrix, student_outputs, test_node_idxs, smoothing=smoothing, **smoothing_kwargs)
    return np.array(test_node_idxs), np.array(student_outputs)
