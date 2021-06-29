import os
import torch
import scipy
import numpy as np
import networkx as nx

from igraph import Graph
from torch_geometric.utils.convert import from_networkx



def getAdjacencyMatrix(G):
	'''Auxiliary function to retrieve the adjacency matrix of a nx.Graph.'''
	return nx.to_numpy_array(G)


def getAdjacencyList(G):
	'''Auxiliary function to retrieve the adjacenty list of a nx.Graph.'''
	adj_list = {k: list(v.keys()) for k, v in nx.to_dict_of_dicts(G).items()}
	return adj_list


def fromNetworkx2GraphML(dataset):
    '''
    Auxiliary function that converts a networkx dataset to graphML data.

    Parameters:
        - dataset: (list<nx.Graph>) List of networkx graphs.

    Returns:
        - (list<igraph.Graph>) Converted data.
    '''
    temp_filename = 'temp_graph.graphml'
    graphml_dataset = []
    for G in dataset:
        # As there is no direct function to convert to graphml on the fly,
        # we do this workaround to convert the data.
        nx.write_graphml(G, temp_filename)
        G = Graph.Read_GraphML(temp_filename)
        graphml_dataset.append(G)
        os.remove(temp_filename)
    return graphml_dataset

    
def fromNetworkx2Torch(dataset, initial_relabeling=None):
    '''
    Auxiliary function that converts a networkx dataset to torch_geometric.data.

    Parameters:
    	- dataset: (list<nx.Graph>) List of networkx graphs.
    	- initial_labeling: (str) Type of initial relabeling to perform (if needed).

    Returns:
    	- (list<torch_geometric.data>) Converted list.
    '''
    torch_dataset = []
    #
    for g in dataset:
        torch_g = from_networkx(g)
        # Add the node degrees as node features
        if initial_relabeling == 'degrees':
        	torch_g.x = torch.Tensor(np.array(list(dict(g.degree()).values()))[:, None])
        elif initial_relabeling == 'ones':
            torch_g.x = torch.Tensor(np.array([1 for _ in range(len(g))])[:, None])
        torch_dataset.append(torch_g)
    return torch_dataset


def addLabels(dataset, labels):
    '''
    Add labels to a pytorch dataset.

    Parameters:
        - dataset: (list<torch_geometric.data>) Dataset to add the labels to.
        - labels: (list<list<float>>) Labels of the dataset.

    Returns:
        - (list<torch_geometric.data>) Final dataset.
    '''
    torch_dataset = []
    for i, g in enumerate(dataset):
        g.y = torch.Tensor(np.array(labels[i])[:, None])
        torch_dataset.append(g)
    return torch_dataset


def fromAPTED2NTK():
    pass


def expandCompressedDistMatrix(dist_matrix):
    '''
    Expand a compressed pairwise distance matrix.

    Parameters:
        - dist_matrix: (np.array) 1/2 * n (n - 1) 1d array with the pairwise distances.

    Returns:
        - (np.ndarray) Full symmetric matrix induced by the input vector.
    '''
    return scipy.spatial.distance.squareform(dist_matrix)
