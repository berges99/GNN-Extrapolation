import numpy as np

from tqdm import tqdm



def computeD(W):
	'''Auxiliary function to compute the diagonal matrix for a given adjacency matrix (weighted).'''
	return np.diag([np.sum(W[i, :]) for i in range(len(W))])


def auxiliaryGaussianKernel(dist, threshold=None, normalization_factor=None):
	'''Auxiliary function that computes the thresholded Gaussian kernel weighting function.'''
	if threshold and threshold > dist:
		return 0
	else:
		if normalization_factor:
			return np.exp(-(dist**2 / normalization_factor))
		else:
			return np.exp(-dist)


def computeW(dist_matrix, threshold=None, normalization_factor=None):
	'''
	Function that computes the induced adjacency matrix W by the pairwise relationships/distances
	between the data points in the dataset.

	Parameters:
		- dist_matrix: (np.ndarray) Input pairwise distances between data points in the dataset.
		- threshold: Threshold for the distances.
		- normalization_factor: Normalization factor for the weighting function.

	Returns:
		- (np.ndarray) Adjacency matrix induced by the pairwise distances.
	'''
	W = np.zeros_like(dist_matrix)
	for i in range(len(W)):
		for j in range(len(W)):
			W[i, j] = auxiliaryGaussianKernel(
				dist_matrix[i, j], threshold=threshold, normalization_factor=normalization_factor)
			W[j, i] = W[i, j]
	return W


def computeLaplacian(W, normalize=False):
	'''
	Function that computes the graph Laplacian, defined as L = D - W.

	Parameters:
		- W: (np.ndarray) Input adjacency matrix.
		- normalize: (bool) Whether to use the normalized graph Laplacian.

	Returns:
		- (np.ndarray) Graph Laplacian of the given adjacency matrix.
	'''
	D = computeD(W)
	L = D - W
	if normalize:
		D_inv_sqrt = np.diag([D[i, i]**(-0.5) for i in range(len(L))])
		L = D_inv_sqrt @ L @ D_inv_sqrt
	return L


def computeSmoothness(L, F):
	'''
	Smoothnesss with respect to the intrinsic structure of the data domain, which in our context is
	the weighted graph. We use the p-Dirichlet form of f with p = 2 (check https://arxiv.org/abs/1211.0053).

	Parameters:
		- L: (np.ndarray) Graph Laplacian matrix.
		- F: (np.ndarray) Outputs/predictions/labels for the given data.
		                  Shape: either (n x 1) for regression or (n x d) for classification.

	Returns:
		- (float) Notion of global smoothness.
	'''
	return np.trace(F.T @ L @ F)
