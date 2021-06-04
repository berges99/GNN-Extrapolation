import os
import re
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from utils.io import *
from utils.data import *
from utils.stats import *
from utils.convert import *
from models.Trees.compute_distances import computeDistMatrix
from models.Trees.compute_trees import computeDatasetRootedTrees



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--filename', '-f', type=str, default='', help='Filename of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/preferential_attachment', help='Default path to the data.')
    parser.add_argument(
    	'--method', '-m', type=str, default='apted', 
    	help='Method/algorithms to use for computing trees and distances between them.')
    parser.add_argument(
    	'--depth', '-d', type=int, default=3, help='Maximum depth of the rooted trees.')
    parser.add_argument(
    	'--relabel', type=bool, default=True, 
    	help='Whether to use relabeling of the nodes, i.e. we do not weight relabeling in the edit distance.')
    return parser.parse_args()


def baseline(dataset_rooted_trees_flatten, 
		     regression_outputs_flatten, 
		     dist_matrix, 
		     train_idxs, 
		     test_idxs,
		     aggregator='mean', 
		     smoothed=True):
	'''
	Implementation of the baseline model. It gives predictions for the test data based on rooted
	trees similarities.

	Parameters:
		- dataset_rooted_trees_flatten: (np.array) List with all the flattened rooted trees in the dataset.
		- regression_outputs_flatten: (np.array) List with all the flattened outputs for all the nodes in the dataset.
		- dist_matrix: (np.ndarray) Pairwise distance matrix between all nodes in the dataset.
		- train_idxs: (np.array) Graphs to be used as train data.
		- test_idxs: (np.array) Graphs to be used as test data.
		- aggregator: (str) Aggregator to use when multiple rooted trees are at the same distance.
		- smoothed: (bool) Whether to use closest trees for the predictions.

	Returns:
		- (np.array) Flattened indices of the predicted values (withing the main dataset).
		- (np.array) Flattened predictions for the test data.
	'''
	aggregator = np.mean if aggregator == 'mean' else np.mean # To be continued
	# Number of nodes per graph
	n = 30
	# Set test rows to infinity, such that we cannot use them for the predictions
	test_node_idxs = []
	for test_graph_idx in test_idxs:
		test_node_idxs.extend([x for x in range(test_graph_idx * n, test_graph_idx * n + n)])
	for test_node_idx in test_node_idxs:
		dist_matrix[test_node_idx, :] = [np.inf] * len(dist_matrix)
	# Get the closest trees
	predictions = []
	for test_node_idx in test_node_idxs:
		min_dist = np.min(dist_matrix[:, test_node_idx])
		# Check if we want exact or fuzzy matching
		if smoothed or min_dist == 0:
			min_idxs = np.where(dist_matrix[:, test_node_idx] == min_dist)[0]
			prediction = aggregator(regression_outputs_flatten[min_idxs])
			predictions.append(prediction)
		else:
			predictions.append(0)
	return np.array(test_node_idxs), np.array(predictions)


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(f'{args.path}/raw')
	# Read the dataset and convert to adjacency lists
	networkx_dataset = readPickle(f'{args.path}/raw/{filename}')
	adj_list_dataset = [getAdjacencyList(G) for G in networkx_dataset]
	
	##########
	# Compute all the d-patterns for every node in the dataset (if necessary)
	if os.path.isfile(f'{args.path}/rooted_trees/{filename}'):
		dataset_rooted_trees = readPickle(f'{args.path}/rooted_trees/{filename}')
	else:
		dataset_rooted_trees = computeDatasetRootedTrees(
			adj_list_dataset, method=args.method, depth=args.depth, parallel=False)
		writePickle(dataset_rooted_trees, f'{args.path}/rooted_trees/{filename}')
	
	##########
	# Compute the pairwise distance matrix if necessary (relabel if indicated)
	if os.path.isfile(f'{args.path}/dist_matrices/{filename}'):
		dataset_rooted_trees_flatten, dist_matrix = readPickle(f'{args.path}/dist_matrices/{filename}')
	else:
		dataset_rooted_trees_flatten = [
			re.sub(r"\d+", 'x', item) if args.relabel else item
			for sublist in dataset_rooted_trees for item in sublist
		]
		dist_matrix = computeDistMatrix(
			dataset_rooted_trees_flatten, method=args.method, nystrom=False, parallel=True)
		writePickle((dataset_rooted_trees_flatten, dist_matrix), f'{args.path}/dist_matrices/{filename}')
	
	##########
	# Compute and store basic dataset stats
	computeDatasetStats(networkx_dataset, dataset_rooted_trees_flatten, dist_matrix, filepath=f'{args.path}/rooted_trees', filename=filename, sample=1)

	##########
	# Read the teacher outputs of the dataset and split the data
	regression_outputs_filename = getLatestVersion(
		f'{args.path}/teacher_outputs', filename=filename.rstrip('.pkl'))
	regression_outputs = readPickle(f'{args.path}/teacher_outputs/{regression_outputs_filename}')
	regression_outputs_flatten = np.array([item for sublist in regression_outputs for item in sublist])
	# Generate random shuffle
	train_idxs, test_idxs = generateShuffle(len(dataset_rooted_trees), sort=True)
	
	

	##########
	# Predict on test data with the baseline method
	test_node_idxs, predictions = baseline(
		dataset_rooted_trees_flatten, regression_outputs_flatten, dist_matrix, train_idxs, test_idxs, smoothed=True)
	print(test_node_idxs, predictions)
	# Evaluate the performance
	total_error, avg_G_error, avg_n_error = evaluatePerformance(predictions, regression_outputs_flatten[test_node_idxs])
	print(total_error, avg_G_error, avg_n_error)
	
	



	
		
	

if __name__ == '__main__':
	main()
