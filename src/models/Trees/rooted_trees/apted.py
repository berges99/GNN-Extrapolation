import re
import numpy as np

from numba import jit



LEAF_PATTERN = re.compile(r"\{\d+\}")


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
        rooted_trees[i] = T # removed relabeling
    return rooted_trees


###


# @jit(nopython=True)
# def computeRootedTreesNumba(adj_list, depth=3):
#     '''

#     TBD

#     '''
#     rooted_trees = np.zeros((len(adj_list), 1200), np.int64) - 3
#     for i in range(len(adj_list)):
#         T = np.array([-1, i, -2], np.int64)
#         leaf_idxs = np.array([1], np.int64)
#         for _ in range(depth):
#             offset = 0
#             leaf_idxs_ = np.zeros(0, np.int64)
#             for j in range(len(leaf_idxs)):
#                 idx = leaf_idxs[j] + offset
#                 neighbors = adj_list[T[idx]]
#                 neighbors = neighbors[np.where(neighbors != -3)]
#                 n = len(neighbors)
#                 new_chunk = np.empty(n + 2 * n, np.int64)
#                 for k in range(n):
#                     new_chunk[3 * k:3 * k + 3] = np.array([-1, neighbors[k], -2], np.int64)
#                 leaf_idxs_ = np.concatenate((leaf_idxs_, np.arange(1, len(new_chunk), 3) + idx + 1))
#                 T = np.concatenate((T[:idx + 1], new_chunk, T[idx + 1:]))
#                 offset += len(new_chunk)
#             leaf_idxs = leaf_idxs_
#         rooted_trees[i,:len(T)] = T
#     return rooted_trees
