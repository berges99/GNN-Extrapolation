import numpy as np



def computeDistance(repr1, repr2):
	'''
	The l2 distance or Euclidean distance between two points in Euclidean space is the 
	length of a line segment between the two points.

	Parameters:
		- repr1, repr2: (iterables, e.g. str or array_like) Representations to compare.

	Returns:
		- (float) Distance between the two given inputs.
	'''
	return np.linalg.norm((repr1 - repr2), ord=2)