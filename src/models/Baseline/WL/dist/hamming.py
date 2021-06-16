import numpy as np



# TODO: apply parameterized downscaling for "further" hash values, e.g. alpha_{i+1} = alpha_i / degree_max
def computeDistance(repr1, repr2):
	'''
	The hamming distance between two inputs of equal length is the number of positions
	at which these inputs vary.

	Parameters:
		- repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

	Returns:
		- (float) Distance between the two given inputs.
	'''
	return np.sum(np.array([c1 != c2 for c1, c2 in zip(repr1, repr2)], dtype=float))