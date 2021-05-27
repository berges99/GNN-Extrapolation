import time
import importlib
import numpy as np
import multiprocessing

from tqdm import tqdm
from joblib import Parallel, delayed



NUM_CORES = multiprocessing.cpu_count()


def computeDatasetRootedTrees(adj_list_dataset, method='apted', depth=3, parallel=False):
    '''
    Compute all the d-patters/rooted-trees for all nodes for all the graphs of the dataset.

    Parameters:
        - dataset: (list<dict<list>>) List with the adjacency lists of every graph in the dataset.
        - depth: (int) Depth of the rooted trees.
        - parallel: (bool) Whether to leverage parallel computation.

    Returns:
        - (list<list<str>>) All rooted trees for every node of every graph in the input dataset.
    '''
    print()
    print('-' * 30)
    print('Computing rooted trees for all nodes in the dataset...')
    # Import the method for compute the edit distance between a pair of trees
    computeRootedTrees = getattr(
        importlib.import_module(f'models.Trees.rooted_trees.{method}'), 'computeRootedTrees')
    if parallel:
        dataset_rooted_trees = Parallel(n_jobs=NUM_CORES)(delayed(computeRootedTrees)(adj_list) for adj_list in adj_list_dataset)
    else:
        dataset_rooted_trees = []
        for adj_list in tqdm(adj_list_dataset):
            rooted_trees = computeRootedTrees(adj_list, depth=depth)
            dataset_rooted_trees.append(rooted_trees)
    # # Numba
    # for i, adj_list in enumerate(tqdm(adj_list_dataset)):
    #     adj_list_dataset[i] = np.array([np.concatenate((np.array(v, dtype=np.int64), np.zeros(100 - len(v), dtype=np.int64) - 3), dtype=np.int64) for _, v in sorted(adj_list.items())], dtype=np.int64)
    #     rooted_trees = computeRootedTreesNumba(adj_list_dataset[i])
    # #results = Parallel(n_jobs=NUM_CORES)(delayed(computeRootedTreesNumba)(adj_list) for adj_list in adj_list_dataset)
    return dataset_rooted_trees
