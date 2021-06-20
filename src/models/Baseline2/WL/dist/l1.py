import numpy as np



def computeDistance(repr1, repr2):
	'''
	The l1 distance is the sum of absolute difference between the measures in all 
	dimensions of two points.

	Parameters:
		- repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

	Returns:
		- (float) Distance between the two given inputs.
	'''
	return np.linalg.norm((repr1 - repr2), ord=1)