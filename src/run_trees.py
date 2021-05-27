import os
import re
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict

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


def train(X_train, y_train):
	'''


	'''
	print()
	print('-' * 30)
	print(f'Hashing seen trees in the training dataset... ' \
		  f'({len(X_train)} graphs, i.e. {len(X_train) * len(X_train[0])} nodes)')
	seen_trees = defaultdict(list)
	# For all the graphs in the dataset
	for i, G in enumerate(tqdm(X_train)):
		# For all the trees in the graph
		for j, T in enumerate(G):
			seen_trees[T].append(y_train[i][j])
	return seen_trees


def test(seen_trees, X_test, agg='mean'):
	'''

	TBD


	'''
	print()
	print('-' * 30)
	print('Utilizing seen trees in the dataset to make predictions for the new data...')
	# Determine the aggregator function (TBC)
	aggregator = np.mean if agg == 'mean' else np.mean
	predictions = []
	# For all the graphs in the test data
	for i, G in enumerate(tqdm(X_test)):
		# For all the trees in the graph
		prediction = []
		for j, T in enumerate(G):
			# Check if we had seen the tree
			if T in seen_trees:
				prediction.append(aggregator(seen_trees[T]))
			else:
				prediction.append(0)
		predictions.append(prediction)
	return predictions


def baseline(dataset_rooted_trees_flatten, regression_outputs_flatten, dist_matrix, train_idxs, test_idxs, smoothed=True):
	'''

	'''
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
		if smoothed:
			min_idxs = np.where(dist_matrix[:, test_node_idx] == min_dist)[0]
			prediction = np.mean(np.array(regression_outputs_flatten)[min_idxs])
			predictions.append(prediction)
		else:
			if min_dist == 0:
				min_idxs = np.where(dist_matrix[:, test_node_idx] == min_dist)[0]
				prediction = np.mean(np.array(regression_outputs_flatten)[min_idxs])
				predictions.append(prediction)
			else:
				predictions.append(0)
	return test_node_idxs, predictions


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
		# unique_trees, dist_matrix = readPickle(f'{args.path}/dist_matrices/{filename}')
		dataset_rooted_trees_flatten, dist_matrix = readPickle(f'{args.path}/dist_matrices/{filename}')
	else:
		# unique_trees = list(set([
		# 	re.sub(r"\d+", 'x', item) if args.relabel else item
		# 	for sublist in dataset_rooted_trees for item in sublist
		# ]))
		# dist_matrix = computeDistMatrix(unique_trees, method=args.method, nystrom=False, parallel=True)
		# writePickle((unique_trees, dist_matrix), f'{args.path}/dist_matrices/{filename}')
		dataset_rooted_trees_flatten = [
			re.sub(r"\d+", 'x', item) if args.relabel else item
			for sublist in dataset_rooted_trees for item in sublist
		]
		dist_matrix = computeDistMatrix(
			dataset_rooted_trees_flatten, method=args.method, nystrom=False, parallel=True)
		writePickle((dataset_rooted_trees_flatten, dist_matrix), f'{args.path}/dist_matrices/{filename}')
	
	##########
	# Read the teacher outputs of the dataset and split the data
	regression_outputs_filename = getLatestVersion(
		f'{args.path}/teacher_outputs', filename=filename.rstrip('.pkl'))
	regression_outputs = readPickle(f'{args.path}/teacher_outputs/{regression_outputs_filename}')
	regression_outputs_flatten = [item for sublist in regression_outputs for item in sublist]
	# Generate random shuffle
	train_idxs, test_idxs = generateShuffle(len(dataset_rooted_trees), sort=True)
	
	


	test_node_idxs, predictions = baseline(
		dataset_rooted_trees_flatten, regression_outputs_flatten, dist_matrix, train_idxs, test_idxs, smoothed=True)

	print(test_node_idxs, predictions)

	# Evaluate the performance
	total_error, avg_G_error, avg_n_error = evaluatePerformance(
		predictions, np.array(regression_outputs_flatten)[test_node_idxs])
	print(total_error, avg_G_error, avg_n_error)
	
	



	#computeStats(0, f'{args.path}/rooted_trees', filename)
	


	##########
	# # Check all seen trees in the dataset
	#X_train, X_test, y_train, y_test = splitData(dataset_rooted_trees, regression_outputs)
	# seen_trees = train(X_train, y_train)
	# # Use the seen trees to predict on the test data
	# predictions = test(seen_trees, X_test)	
	

if __name__ == '__main__':
	main()
