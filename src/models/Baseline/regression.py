import numpy as np



def baseline(node_representations_flatten,
             node_representations_idxs,
             teacher_outputs_flatten, 
             dist_matrix, 
             train_idxs, 
             test_idxs=None,
             smoothing='none'):
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
        - smoothing: (str) Which type of smoothing to apply.

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
    np.fill_diagonal(dist_matrix, np.inf)
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
    # Get the closest trees
    student_outputs = []
    for test_node_idx in test_node_idxs:
        min_dist = np.min(dist_matrix[:, test_node_idx])
        # Check if we want exact or fuzzy matching
        if smoothing == 'knn' or min_dist == 0:
            min_idxs = np.where(dist_matrix[:, test_node_idx] == min_dist)[0]
            prediction = np.mean(teacher_outputs_flatten[min_idxs])
            student_outputs.append(prediction)
        else:
            student_outputs.append(np.mean(teacher_outputs_flatten))
    return np.array(test_node_idxs), np.array(student_outputs)
