import torch
import pickle
import numpy as np

from torch_geometric.utils.convert import from_networkx



def writePickleNetworkx(dataset, filename):
	'''Auxiliary function that writes a list of networkx graphs as serialized python objects.'''
	with open(filename, 'wb') as f:
		pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def readPickleNetworkx(filename):
	'''Auxiliary function that writes a list of networkx graphs as serialized python objects.'''
	with open(filename, 'rb') as f:
		return pickle.load(f)


def fromNetworkx(dataset, add_degree=True):
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
        torch_dataset.append(torch_g)
    return torch_dataset
