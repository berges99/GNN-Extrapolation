import numpy as np



def splitData(dataset, outputs=None, train_size=0.8, seed=None):
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
			np.array(dataset)[train_idxs], 
			np.array(dataset)[test_idxs], 
			np.array(outputs)[train_idxs], 
			np.array(outputs)[test_idxs]
		)
	else:
		return (
			np.array(dataset)[train_idxs], 
			np.array(dataset)[test_idxs]
		)
