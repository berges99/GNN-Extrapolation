import os
import re
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from collections import OrderedDict

from utils.data import generateShuffle
from utils.stats import evaluatePerformance
from utils.io import readPickle, writePickle, getLatestVersion
from utils.convert import getAdjacencyList, fromNetworkx2Torch, fromNetworkx2GraphML

from models.Baseline.compute_distances import computeDistMatrix
from models.Baseline.compute_node_representations import computeDatasetNodeRepresentations



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    # 	'--fullname', '-f', type=str, default='', help='Complete relative path of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/preferential_attachment', help='Default path to the data.')
    # parser.add_argument(
    # 	'--setting', '-s', type=str, default='regression', choiches=['regression', 'classification'],
    # 	help='Whether to ')
    ###
    parser.add_argument(
    	'--depth', '-d', type=int, default=3, 
    	help='Max. receptive field depth for extracting the node representations, e.g. depth of the rooted trees.')
    ###
    subparsers = parser.add_subparsers(dest='embedding_scheme')
    subparsers.required = True
    ###
    WL = subparsers.add_parser('WL', help='WL kernel embedding_scheme parser.')
    WL.add_argument(
    	'--method', type=str, default='continuous', choices=['continuous', 'hashing'],
    	help='Method to compute the WL kernel. Available choices are [continuous, categorical].')
    WL.add_argument(
    	'--normalization', type=str, default='wasserstein', choices=['wasserstein', 'GCN'],
    	help='Normalization to apply at each step of the WL kernel. Available choices are [wasserstein, GCN].')
    WL.add_argument(
    	'--distance', type=str, default='hamming', choices=['hamming', 'l1', 'l2'],
    	help='Distance to use for computing the distances between node representations. Available choices are [hamming, l1, l2].')
    ###
    rooted_trees = subparsers.add_parser('Trees', help='Rooted trees embedding_scheme parser.')
    rooted_trees.add_argument(
    	'--method', type=str, default='apted', choices=['apted'],
    	help='Method to use for the extraction of rooted trees/d-patterns. Available choices are [apted].')
    rooted_trees.add_argument(
    	'--distance', type=str, default='apted', choices=['apted'],
    	help='Distance to use for computing the distances/kernel values between rooted trees. Available choices are [apted].')
    return parser.parse_args()


def baseline(node_representations_flatten,
			 node_representations_idxs,
		     regression_outputs_flatten, 
		     dist_matrix, 
		     train_idxs, 
		     test_idxs,
		     aggregator='mean', 
		     smoothed=True):
	'''
	Implementation of the baseline model. It gives predictions for the test data based on node representation
	similarities.

	Parameters:
		- node_representations_flatten: (array_like) List with all the flattened node representations in the dataset.
		- node_representations_idxs: (np.array<int>) Array with the number of nodes per graph in the dataset.
		- regression_outputs_flatten: (np.array<float>) Array with all the flattened outputs for all the nodes in the dataset.
		- dist_matrix: (np.ndarray) Pairwise distance matrix between all nodes in the dataset.
		- train_idxs: (np.array) Graphs to be used as train data.
		- test_idxs: (np.array) Graphs to be used as test data.
		- aggregator: (str) Aggregator to use when multiple rooted trees are at the same distance.
		- smoothed: (bool) Whether to use closest trees for the predictions.

	Returns:
		- (np.array) Flattened indices of the predicted values (within the main dataset).
		- (np.array) Flattened predictions for the test data.
	'''
	aggregator = np.mean if aggregator == 'mean' else np.mean # TBD
	# Set test rows to infinity, such that we cannot use them for the predictions
	test_node_idxs = []
	for test_graph_idx in test_idxs:
		n = node_representations_idxs[test_graph_idx]
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

	# Read the raw networkx dataset
	dataset_filename = getLatestVersion(f'{args.path}/raw')
	networkx_dataset = readPickle(f'{args.path}/raw/{dataset_filename}')
	# Fromat the data in a convenient way
	if args.embedding_scheme == 'WL':
		if args.method == 'continuous':
			formatted_dataset = fromNetworkx2Torch(networkx_dataset, add_degree=True)
		else: # elif args.method == 'hashing'
			formatted_dataset = fromNetworkx2GraphML(networkx_dataset)
	else: # elif args.embedding_scheme == 'Trees'
		formatted_dataset = [getAdjacencyList(G) for G in networkx_dataset]
	
	##########
	# Compute all the node representations for every node in the dataset # (if necessary)
	node_representations_kwargs = OrderedDict({
		k: v for k, v in vars(args).items() 
		if k not in ['path', 'distance', 'embedding_scheme', 'method']
	})
	node_representations_filename = \
		f"{args.path}/node_representations/{args.embedding_scheme}/{args.method}/" \
		f"{'_'.join([k[0] + str(v).capitalize() for k, v in node_representations_kwargs.items()])}" \
		f"_{dataset_filename}"
	if os.path.isfile(node_representations_filename):
		node_representations = readPickle(node_representations_filename)
	else:
		node_representations = computeDatasetNodeRepresentations(
			formatted_dataset, args.embedding_scheme, args.method, 
			parallel=False, **node_representations_kwargs
		)
		#writePickle(node_representations, filename=node_representations_filename)
	print(node_representations)
	

	# from models.Baseline.WL.repr.hashing import computeNodeRepresentations
	# for i in range(len(formatted_dataset)):
	# 	node_representations = computeNodeRepresentations(formatted_dataset[i], depth=args.depth)
	# 	print(node_representations)
	# return


	##########
	# Compute the pairwise distance matrix if necessary
	distance_kwargs = OrderedDict({'depth': args.depth})
	distances_filename = \
		f"{'/'.join(node_representations_filename.split('/')[:-1])}" \
		f"/dist_matrices/{args.distance}/{node_representations_filename.split('/')[-1]}"
	if os.path.isfile(distances_filename):
		node_representations_flatten, dist_matrix = readPickle(distances_filename)
	else:
		node_representations_flatten = [item for sublist in node_representations for item in sublist]
		dist_matrix = computeDistMatrix(
			node_representations_flatten, args.embedding_scheme, args.distance, 
			nystrom=False, parallel=True, **distance_kwargs
		)
		#writePickle((node_representations_flatten, dist_matrix), filename=distances_filename)
	print(dist_matrix)
	
	# ##########
	# # Compute and store basic dataset stats
	# computeDatasetStats(networkx_dataset, dataset_rooted_trees_flatten, dist_matrix, filepath=f'{args.path}/rooted_trees', filename=filename, sample=1)

	##########
	# Read the teacher outputs of the dataset and split the data
	regression_outputs_filename = getLatestVersion(
		f'{args.path}/teacher_outputs', filename=dataset_filename.rstrip('.pkl'))
	regression_outputs = readPickle(f'{args.path}/teacher_outputs/{regression_outputs_filename}')
	regression_outputs_flatten = np.array([item for sublist in regression_outputs for item in sublist])
	# Generate random shuffle
	train_idxs, test_idxs = generateShuffle(len(node_representations), sort=True)

	##########
	# Predict on test data with the baseline method
	# Compute number of nodes per graph (to handle multiple sized graphs in the future)
	node_representations_idxs = np.array([len(G) for G in node_representations], dtype=int)
	test_node_idxs, predictions = baseline(
		node_representations_flatten, node_representations_idxs, regression_outputs_flatten, dist_matrix, 
		train_idxs, test_idxs, aggregator='mean', smoothed=True
	)
	print(test_node_idxs)
	print(predictions)
	
	# Evaluate the performance
	total_error, avg_G_error, avg_n_error = evaluatePerformance(predictions, regression_outputs_flatten[test_node_idxs])
	print(total_error, avg_G_error, avg_n_error)
	
	



if __name__ == '__main__':
	main()
