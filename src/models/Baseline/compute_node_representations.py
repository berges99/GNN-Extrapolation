import time
import importlib
import numpy as np
import multiprocessing

from tqdm import tqdm
from torch_geometric.data import DataLoader



def computeDatasetNodeRepresentations(formatted_dataset, embedding_scheme,
                                      method, **node_representations_kwargs):
    '''
    Computes all the node representations for all the graphs in the given dataset.

    Parameters:
        - formatted_dataset: (array_like) Array with all the graphs in the dataset formatted
                                          as required by the specified method.
        - embedding_scheme: (str) Embedding scheme to be used, e.g. WL or Trees.
        - method: (str) Concrete method/implementation to use for the chosen scheme.
        - (**node_representations_kwargs) Additional specific arguments for the chosen embedding 
                                          scheme and method.

    Returns:
        - (array_like) 
    '''
    print()
    print('-' * 30)
    print('Computing rooted trees for all nodes in the dataset...')
    # Import the adequate modules for computing the node representations
    computeNodeRepresentations = getattr(importlib.import_module(
        f"models.Baseline.{embedding_scheme}.{method}"), 'computeNodeRepresentations')
    # Prepare the data loader if we are dealing with torch data
    if embedding_scheme == 'WL':
        formatted_dataset = DataLoader(formatted_dataset, batch_size=1)
    dataset_node_representations = []
    for G in tqdm(formatted_dataset):
        node_representations = computeNodeRepresentations(G, **node_representations_kwargs)
        dataset_node_representations.append(node_representations)
    return dataset_node_representations
