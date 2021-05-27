import re

from apted import APTED
from apted.helpers import Tree



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
