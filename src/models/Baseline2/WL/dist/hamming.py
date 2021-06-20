import numpy as np



def computeDistance(repr1, repr2, scaling=None):
	'''
	The hamming distance between two inputs of equal length is the number of positions
	at which these inputs vary.

	Parameters:
		- repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

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
