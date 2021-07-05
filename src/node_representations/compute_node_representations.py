import time
import importlib
import numpy as np
import multiprocessing

from tqdm import tqdm
from joblib import Parallel, delayed
from torch_geometric.data import DataLoader



NUM_CORES = multiprocessing.cpu_count()


def computeDatasetNodeRepresentations(formatted_dataset, 
                                      embedding_scheme, 
                                      method, 
                                      parallel=False, 
                                      **node_representations_kwargs):
    '''
    Computes all the node representations for all the graphs in the given dataset.

    Parameters:
        - formatted_dataset: (array_like) Array with all the graphs in the dataset formatted
                                          as required by the specified method.
        - embedding_scheme: (str) Embedding scheme to be used, e.g. WL or Trees.
        - method: (str) Concrete method/implementation to use for the chosen scheme.
        - parallel: (bool) Whether to compute node representations using all cores available.
        - (**node_representations_kwargs) Additional specific arguments for the chosen embedding 
                                          scheme and method.

    Returns:
        - (array_like) Returns all the node representations in the dataset.
                       (Same shape as input).
    '''
    print()
    print('Computing rooted trees for all nodes in the dataset...')
    print('-' * 30)
    # Import the adequate modules for computing the node representations
    computeNodeRepresentations = getattr(importlib.import_module(
        f"node_representations.{embedding_scheme}.{method}"), 'computeNodeRepresentations')
    # Compute all the embeddings with all the dataset when we are using the WL kernel hashing
    if embedding_scheme == 'WL' and method == 'hashing':
        dataset_node_representations = computeNodeRepresentations(formatted_dataset, **node_representations_kwargs)
    else:
        # Prepare the data loader if we are dealing with torch data
        if embedding_scheme == 'WL' and method == 'continuous':
            formatted_dataset = DataLoader(formatted_dataset, batch_size=1)
        # Parallelize computations at graph level
        if parallel:
            dataset_node_representations = \
                (Parallel(n_jobs=NUM_CORES)
                         (delayed(computeNodeRepresentations)(G) for G in tqdm(formatted_dataset)))   
        else:
            dataset_node_representations = []
            for G in tqdm(formatted_dataset):
                node_representations = computeNodeRepresentations(G, **node_representations_kwargs)
                dataset_node_representations.append(node_representations)
    return dataset_node_representations
