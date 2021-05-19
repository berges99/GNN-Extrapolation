import torch
import numpy as np

from torch_geometric.data import Data



def splitData(dataset, outputs=None, train_size=0.9, seed=None):
	'''
	Generate random train/test splitting.

	Parameters:
		- dataset: (list<obj>) List of X data.
		- outputs: (list<obj>) List of y data.
		- train_size: (float) Proportion of elements to train with.
		- seed: (int) Whether to set a random seed for reproducibility reasons.

	Returns:
		- (np.array) X_train, X_test, y_train, y_test
	'''
	n = len(dataset)
	if seed is not None:
		np.random.seed(seed)
	shuffle = np.arange(n)
	np.random.shuffle(shuffle)
	train_idxs, test_idxs = shuffle[:int(train_size * n)], shuffle[int(train_size * n):]
	if outputs:
		return (
			np.array(dataset, dtype=object)[train_idxs], 
			np.array(dataset, dtype=object)[test_idxs], 
			np.array(outputs, dtype=object)[train_idxs], 
			np.array(outputs, dtype=object)[test_idxs]
		)
	else:
		return (
			# ('edge_index', 'x', 'y')
			[Data(x=X[1][1], edge_index=X[0][1], y=X[2][1]) for X in np.array(dataset, dtype=object)[train_idxs]],
			[Data(x=X[1][1], edge_index=X[0][1], y=X[2][1]) for X in np.array(dataset, dtype=object)[test_idxs]]
		)
