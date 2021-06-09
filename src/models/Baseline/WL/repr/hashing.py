import copy
import numpy as np
import igraph as ig

from typing import List
from sklearn.preprocessing import scale
from sklearn.base import TransformerMixin



class WeisfeilerLehman(TransformerMixin):
    """
    Class that implements the Weisfeiler-Lehman transform.
    
    Modified version of "Christian Bock and Bastian Rieck" implementation, so that
    it computes the representations one graph at a time, aiming towards parallelization
    i.e. faster computational times.
    """
    def __init__(self):
        self._relabel_steps = {}
        self._label_dict = {}
        self._last_new_label = -1
        self._preprocess_relabel_dict = {}
        self._results = {}
        self._label_dicts = {}

    def _reset_label_generation(self):
        self._last_new_label = -1

    def _get_next_label(self):
        self._last_new_label += 1
        return self._last_new_label

    def _preprocess_graph(self, g: ig.Graph):
        num_unique_labels = 0
        x = g.copy()
        # If the graph nodes are unlabeled use the degree of each node
        if not 'label' in x.vs.attribute_names():
            x.vs['label'] = list(map(str, [l for l in x.vs.degree()]))           
        labels = x.vs['label']
        # Relabel
        new_labels = []
        for label in labels:
            if label in self._preprocess_relabel_dict.keys():
                new_labels.append(self._preprocess_relabel_dict[label])
            else:
                self._preprocess_relabel_dict[label] = self._get_next_label()
                new_labels.append(self._preprocess_relabel_dict[label])
        x.vs['label'] = new_labels
        self._results[0] = (labels, new_labels)
        self._label_sequences[:, 0] = new_labels
        self._reset_label_generation()
        return x

    def fit_transform(self, g: ig.Graph, num_iterations: int=3, return_sequences=True):
        self._label_sequences = np.full((len(g.vs), num_iterations + 1), np.nan)
        g = self._preprocess_graph(g)
        for it in range(1, num_iterations + 1):
            self._reset_label_generation()
            self._label_dict = {}
            # Get labels of current iteration
            current_labels = g.vs['label']
            # Get for each vertex the labels of its neighbors
            neighbor_labels = self._get_neighbor_labels(g, sort=True)
            # Prepend the vertex label to the list of labels of its neighbors
            merged_labels = [[b] + a for a, b in zip(neighbor_labels, current_labels)]
            # Generate a label dictionary based on the merged labels
            self._append_label_dict(merged_labels)
            # Relabel the graph
            new_labels = self._relabel_graph(g, merged_labels)
            self._relabel_steps[it] = {
                idx: {old_label: new_labels[idx]} 
                for idx, old_label in enumerate(current_labels)
            }
            g.vs['label'] = new_labels
            self._results[it] = (merged_labels, new_labels)
            self._label_sequences[:, it] = new_labels
            self._label_dicts[it] = copy.deepcopy(self._label_dict)
        if return_sequences:
            return self._label_sequences
        else:
            return self._results

    def _relabel_graph(self, X: ig.Graph, merged_labels: list):
        new_labels = []
        for merged in merged_labels:
            new_labels.append(self._label_dict['-'.join(map(str, merged))])
        return new_labels

    def _append_label_dict(self, merged_labels: List[list]):
        for merged_label in merged_labels:
            dict_key = '-'.join(map(str,merged_label))
            if dict_key not in self._label_dict.keys():
                self._label_dict[ dict_key ] = self._get_next_label()

    def _get_neighbor_labels(self, X: ig.Graph, sort: bool=True):
            neighbor_indices = [[n_v.index for n_v in X.vs[X.neighbors(v.index)]] for v in X.vs]
            neighbor_labels = []
            for n_indices in neighbor_indices:
                if sort:
                    neighbor_labels.append( sorted(X.vs[n_indices]['label']) )
                else:
                    neighbor_labels.append( X.vs[n_indices]['label'] )
            return neighbor_labels


def computeNodeRepresentations(G, depth=3, **kwargs):
    '''
    Compute the WL kernel nodde representations using the hashing version.

    Parameters:
        - G: (ig.Graph) Input graph in graphml format.
        - depth: (int) Number of aggregation steps to perform.

    Returns:
        - (array_like) Node representations for every node in the input graph.
    '''
    wl = WeisfeilerLehman()
    node_representations = wl.fit_transform(G, num_iterations=depth)
    return node_representations
