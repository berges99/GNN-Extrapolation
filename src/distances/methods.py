import re
import numpy as np

from numba import jit
from apted import APTED
from apted.helpers import Tree



def edit(T1, T2, relabel=True):
    '''
    Compute the edit distance between two trees written in bracket format.

    Parameters:
        - T1, T2: (str) Input trees in bracket format.

    Returns:
        - (int) Edit distance between the two input trees.
    '''
    if relabel:
        T1 = re.sub(r"\d+", 'x', T1)
        T2 = re.sub(r"\d+", 'x', T2)
    T1 = Tree.from_text(T1)
    T2 = Tree.from_text(T2)
    apted = APTED(T1, T2)
    return apted.compute_edit_distance()


##########


def hamming(repr1, repr2, scaling=None):
    '''
    The hamming distance between two inputs of equal length is the number of positions
    at which these inputs vary.

    Parameters:
        - repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.
        - scaling: (float) Scaling to apply at each level of the hamming distance.

    Returns:
        - (float) Distance between the two given inputs.
    '''
    if scaling:
        alphas = [1]
        for i in range(1, len(repr1)):
            alphas.append(alphas[i - 1] / scaling)
    # If not specified apply no scaling
    else:
        alphas = np.ones_like(repr1)
    return np.sum(np.array([
        alphas[i] * (c1 != c2) for i, (c1, c2) in enumerate(zip(repr1, repr2))], dtype=float))


@jit(nopython=True)
def hammingNumba(repr1, repr2, scaling=0):
    '''
    The hamming distance between two inputs of equal length is the number of positions
    at which these inputs vary.

    Parameters:
        - repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.
        - scaling: (float) Scaling to apply at each level of the hamming distance.

    Returns:
        - (float) Distance between the two given inputs.
    '''
    n = len(repr1)
    alphas = np.ones(n)
    if scaling:
        for i in range(1, n):
            alphas[i] = alphas[i - 1] / scaling
    return alphas @ np.where(repr1 != repr2, 1.0, 0.0)


##########


def l1(repr1, repr2):
    '''
    The l1 distance is the sum of absolute difference between the measures in all 
    dimensions of two points.

    Parameters:
        - repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

    Returns:
        - (float) Distance between the two given inputs.
    '''
    return np.linalg.norm((repr1 - repr2), ord=1)

@jit(nopython=True)
def l1Numba(repr1, repr2):
    '''
    The l1 distance is the sum of absolute difference between the measures in all 
    dimensions of two points.

    Parameters:
        - repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

    Returns:
        - (float) Distance between the two given inputs.
    '''
    return np.linalg.norm((repr1 - repr2), ord=1)


##########


def l2(repr1, repr2):
    '''
    The l2 distance or Euclidean distance between two points in Euclidean space is the 
    length of a line segment between the two points.

    Parameters:
        - repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

    Returns:
        - (float) Distance between the two given inputs.
    '''
    return np.linalg.norm((repr1 - repr2), ord=2)


@jit(nopython=True)
def l2Numba(repr1, repr2):
    '''
    The l2 distance or Euclidean distance between two points in Euclidean space is the 
    length of a line segment between the two points.

    Parameters:
        - repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

    Returns:
        - (float) Distance between the two given inputs.
    '''
    return np.linalg.norm((repr1 - repr2), ord=2)
