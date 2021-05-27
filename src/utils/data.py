import torch
import numpy as np

from torch_geometric.data import Data



def generateShuffle(n, train_size=0.9, seed=None, sort=False):
	'''
	Generate random shuffle of indices for train/test splitting.

	Parameters:
		- train_size: (float) Proportion of elements to train with.
		- seed: (int) Whether to set a random seed for reproducibility reasons.
		- sort: (bool) Whether to return the shuffle sorted or not

	Returns:
		- (np.array) train_idxs, test_idxs
	'''
	# Set random seed if specified
	if seed:
		np.random.seed(seed)
	# Shuffle the collection of data
	shuffle = np.arange(n)
	np.random.shuffle(shuffle)
	train_idxs, test_idxs = shuffle[:int(train_size * n)], shuffle[int(train_size * n):]
	# Return sorted if indicated
	if sort:
		train_idxs = np.sort(train_idxs)
		test_idxs = np.sort(test_idxs)
	return train_idxs, test_idxs


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
	# Generate random shuffle
	train_idxs, test_idxs = generateShuffle(n, train_size, seed, sort=False)
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
