import re

from apted import APTED
from apted.helpers import Tree



def computeDistance(T1, T2, relabel=True):
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
