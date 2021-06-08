import torch
import numpy as np
import networkx as nx

from torch_geometric.utils.convert import from_networkx



def getAdjacencyMatrix(G):
	'''Auxiliary function to retrieve the adjacency matrix of a nx.Graph.'''
	return nx.to_numpy_array(G)


def getAdjacencyList(G):
	'''Auxiliary function to retrieve the adjacenty list of a nx.Graph.'''
	adj_list = {k: list(v.keys()) for k, v in nx.to_dict_of_dicts(G).items()}
	return adj_list


def fromNetworkx2Torch(dataset, add_degree=True):
    '''
    Auxiliary function that converts a networkx dataset to torch_geometric.data.

    Parameters:
    	- dataset: (list<nx.Graph>) List of networkx graphs.
    	- add_degree: (bool) Whether to add the degree for the node features.

    Returns:
    	- (list<torch_geometric.data>) Converted list to 
    '''
    torch_dataset = []
    #
    for g in dataset:
        torch_g = from_networkx(g)
        # Add the node degrees as node features
        if add_degree:
        	torch_g.x = torch.Tensor(np.array(list(dict(g.degree()).values()))[:, None])
        # Otherwise just add a 1.
        else:
            torch_g.x = torch.ones((1, torch_g.size(0)))
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
