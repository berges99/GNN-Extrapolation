import re
import time
import multiprocessing
import numpy as np

from tqdm import tqdm
from numba import jit
from joblib import Parallel, delayed

from apted import APTED
from apted.helpers import Tree



LEAF_PATTERN = re.compile(r"\{\d+\}")
NUM_CORES = multiprocessing.cpu_count()


def computeEditDistance(T1, T2):
    '''
    Compute the edit distance between two trees written in bracket format.

    Parameters:
        - T1, T2: (str) Input trees in bracket format.

    Returns:
        - (int) Edit distance between the two input trees.
    '''
    T1 = Tree.from_text(T1)
    T2 = Tree.from_text(T2)
    apted = APTED(T1, T2)
    return apted.compute_edit_distance()


def computeDistMatrix(trees):
    '''
    Compute pairwise distances between all input trees.

    Parameters:
        - trees: (list<obj>) List with trees written in bracket format.

    Returns:
        - (np.ndarray) Matrix with the computed pairwise distanes.
    '''
    n = len(trees)
    distances = np.zeros((n, n))
    # Symetric matrix
    for i in tqdm(range(n)):
        for j in range(i, n):
            distances[i, j] = computeEditDistance(trees[i], trees[j])
            distances[j, i] = distances[i, j]
    return distances


def findLeaves(T):
    '''
    Find the leaves of a tree written in bracket format.

    Parameters:
        - T: (str) Input tree in bracket format.

    Returns:
        - (list<tuple<int, int>>) List with all the leaves in the tree. 
                                  First position is the leaf, and second is the position
                                  in the string.
    '''
    return [
        (int(m.group(0).strip('{}')), m.span(0))
        for m in re.finditer(LEAF_PATTERN, T)
    ]


def computeRootedTrees(adj_list, depth=3):
    '''
    Compute all the dpatterns for every node of an input graph.

    Parameters:
        - adj_list: (dict<list>) Adjacency list of the input graph.
        - depth: (int) Depth of the rooted trees.

    Returns:
        - (list<str>) List with all the rooted trees of all nodes in the dataset.
                      The trees are written in bracket format, e.g. 
                      
                      - {A{B{X}{Y}{F}}{C}} would represent the following tree:
                                            
                                                A
                                               / \
                                              B   C
                                             /|\
                                            X Y F

    '''
    rooted_trees = ['{' + str(i) + '}' for i in range(len(adj_list))]
    for i, T in enumerate(rooted_trees):
        for d in range(depth):
            offset = 0
            for l in findLeaves(T):
                if len(adj_list[l[0]]):
                    substring = '{' + '}{'.join([str(n) for n in adj_list[l[0]]]) + '}'
                    T = T[:offset + l[1][1] - 1] + substring + T[offset + l[1][1] - 1:]
                    offset += len(substring)
        rooted_trees[i] = re.sub(r"\d+", 'x', T)
    return rooted_trees


###


@jit(nopython=True)
def computeRootedTreesNumba(adj_list, depth=3):
    '''

    TBD

    '''
    rooted_trees = np.zeros((len(adj_list), 1200), np.int64) - 3
    for i in range(len(adj_list)):
        T = np.array([-1, i, -2], np.int64)
        leaf_idxs = np.array([1], np.int64)
        for _ in range(depth):
            offset = 0
            leaf_idxs_ = np.zeros(0, np.int64)
            for j in range(len(leaf_idxs)):
                idx = leaf_idxs[j] + offset
                neighbors = adj_list[T[idx]]
                neighbors = neighbors[np.where(neighbors != -3)]
                n = len(neighbors)
                new_chunk = np.empty(n + 2 * n, np.int64)
                for k in range(n):
                    new_chunk[3 * k:3 * k + 3] = np.array([-1, neighbors[k], -2], np.int64)
                leaf_idxs_ = np.concatenate((leaf_idxs_, np.arange(1, len(new_chunk), 3) + idx + 1))
                T = np.concatenate((T[:idx + 1], new_chunk, T[idx + 1:]))
                offset += len(new_chunk)
            leaf_idxs = leaf_idxs_
        rooted_trees[i,:len(T)] = T
    return rooted_trees


###


def computeDatasetRootedTrees(adj_list_dataset, depth=3, parallel=False):
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
    if parallel:
        dataset_rooted_trees = Parallel(n_jobs=NUM_CORES)(delayed(computeRootedTrees)(adj_list) for adj_list in adj_list_dataset)
    else:
        dataset_rooted_trees = []
        for adj_list in tqdm(adj_list_dataset):
            rooted_trees = computeRootedTrees(adj_list, depth=depth)
            dataset_rooted_trees.append(rooted_trees)
    # for i, adj_list in enumerate(tqdm(adj_list_dataset)):
    #     adj_list_dataset[i] = np.array([np.concatenate((np.array(v, dtype=np.int64), np.zeros(100 - len(v), dtype=np.int64) - 3), dtype=np.int64) for _, v in sorted(adj_list.items())], dtype=np.int64)
    #     rooted_trees = computeRootedTreesNumba(adj_list_dataset[i])
    # #results = Parallel(n_jobs=NUM_CORES)(delayed(computeRootedTreesNumba)(adj_list) for adj_list in adj_list_dataset)
    return dataset_rooted_trees






