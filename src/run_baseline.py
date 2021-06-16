import os
import sys
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from collections import OrderedDict

from utils.data import generateShuffle
from utils.stats import evaluatePerformance
from utils.io import KeepOrderAction, readPickle, writePickle, getLatestVersion
from utils.convert import getAdjacencyList, fromNetworkx2Torch, fromNetworkx2GraphML

from models.Baseline.compute_distances import computeDistMatrix
from models.Baseline.compute_node_representations import computeDatasetNodeRepresentations



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    ##########
	# Parse all the input dataset related arguments
    # parser.add_argument(
    # 	'--fullname', '-f', type=str, default='', help='Complete relative path of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/preferential_attachment',
    	help='Default path to the data.')
    # parser.add_argument(
    # 	'--setting', '-s', type=str, default='regression', choiches=['regression', 'classification'],
    # 	help='Whether to ')
    ##########
	# Parse all the arguments related to the embedding scheme, methods and distances
    subparsers = parser.add_subparsers(dest='embedding_scheme')
    subparsers.required = True
    ###
    WL = subparsers.add_parser('WL', help='WL kernel embedding_scheme parser.')
    WL.add_argument(
    	'--method', type=str, required=True, choices=['continuous', 'hashing'], action=KeepOrderAction, #default='continuous',
    	help='Method to compute the WL kernel. Available choices are [continuous, categorical].')
    WL.add_argument(
    	'--depth', '-d', type=int, required=True, action=KeepOrderAction, #default=3,
    	help='Max. receptive field depth for extracting the node representations, e.g. depth of the rooted trees.')
    WL.add_argument(
    	'--initial_labeling', type=str, required=True, choices=['ones', 'degrees'], action=KeepOrderAction, #default=ones
    	help='Type of labeling to be used in the case that there aren\'t any available. Available choices are [ones, degrees].')
    WL.add_argument(
    	'--normalization', type=str, required='--method continuous' in ' '.join(sys.argv),
    	choices=['wasserstein', 'GCN'], action=KeepOrderAction, #default='wasserstein',
    	help='Normalization to apply at each step of the WL kernel. Available choices are [wasserstein, GCN].')
    WL.add_argument(
    	'--distance', type=str, required=True, choices=['hamming', 'l1', 'l2'], action=KeepOrderAction, #default='hamming',
    	help='Distance to use for computing the distances between node representations. Available choices are [hamming, l1, l2].')
    ###
    rooted_trees = subparsers.add_parser('Trees', help='Rooted trees embedding_scheme parser.')
    rooted_trees.add_argument(
    	'--method', type=str, required=True, choices=['apted'], action=KeepOrderAction, #default='apted',
    	help='Method to use for the extraction of rooted trees/d-patterns. Available choices are [apted].')
    rooted_trees.add_argument(
    	'--depth', '-d', type=int, required=True, action=KeepOrderAction, #default=3,
    	help='Max. receptive field depth for extracting the node representations, e.g. depth of the rooted trees.')
    rooted_trees.add_argument(
    	'--distance', type=str, required=True, choices=['edit'], action=KeepOrderAction, #default='edit',
    	help='Distance to use for computing the distances/kernel values between rooted trees. Available choices are [edit].')
    rooted_trees.add_argument(
    	'--relabel', type=bool, required='--distance edit' in ' '.join(sys.argv), action=KeepOrderAction,
    	help='Whether to perform relabeling of the extracted rooted trees of the dataset, i.e. no relabel cost in edit distance.')
    
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
			# TODO: adapt to classification setting
			predictions.append(np.mean(regression_outputs_flatten))
	return np.array(test_node_idxs), np.array(predictions)


def main():
	args = readArguments()

	# Read the raw networkx dataset
	dataset_filename = getLatestVersion(f'{args.path}/raw')
	networkx_dataset = readPickle(f'{args.path}/raw/{dataset_filename}')
	# Fromat the data in a convenient way
	if args.embedding_scheme == 'WL':
		if args.method == 'continuous':
			formatted_dataset = fromNetworkx2Torch(networkx_dataset, add_degree=True) # canviar per adaptar-ho al nou argument de tipus de initial labeling
		else: # elif args.method == 'hashing'
			formatted_dataset = fromNetworkx2GraphML(networkx_dataset)
	else: # elif args.embedding_scheme == 'Trees'
		formatted_dataset = [getAdjacencyList(G) for G in networkx_dataset]
	
	##########
	# Compute all the node representations for every node in the dataset # (if necessary)
	ordered_args = np.array([x[0] for x in args.ordered_args])
	node_representations_kwargs_idx1 = np.where(ordered_args == 'method')[0][0] + 1
	node_representations_kwargs_idx2 = np.where(ordered_args == 'distance')[0][0]
	node_representations_kwargs = OrderedDict({
		k: v for k, v in args.ordered_args[node_representations_kwargs_idx1:node_representations_kwargs_idx2]
	})
	node_representations_filename = \
		f"{args.path}/node_representations/{args.embedding_scheme}/{args.method}/" \
		f"{'_'.join([k[0] + str(v).capitalize() for k, v in node_representations_kwargs.items()])}" \
		f"__{dataset_filename}"
	if os.path.isfile(node_representations_filename):
		node_representations = readPickle(node_representations_filename)
	else:
		node_representations = computeDatasetNodeRepresentations(
			formatted_dataset, args.embedding_scheme, args.method, 
			parallel=False, **node_representations_kwargs
		)
		writePickle(node_representations, filename=node_representations_filename)

	print()
	print('Node representations:')
	print('-' * 30)
	print(node_representations)

	##########
	# Compute the pairwise distance matrix if necessary
	distance_kwargs = OrderedDict({
		k: v for k, v in args.ordered_args[node_representations_kwargs_idx2:]
	})
	distances_filename = \
		f"{'/'.join(node_representations_filename.split('/')[:-1])}" \
		f"/dist_matrices/{args.distance}/" \
		f"{node_representations_filename.split('__')[0].split('/')[-1]}" \
		f"__{'_'.join([k[0] + str(v).capitalize() for k, v in distance_kwargs.items()])}" \
		f"__{node_representations_filename.split('__')[1]}"
	if os.path.isfile(distances_filename):
		node_representations_flatten, dist_matrix = readPickle(distances_filename)
	else:
		node_representations_flatten = [item for sublist in node_representations for item in sublist]
		dist_matrix = computeDistMatrix(
			node_representations_flatten, args.embedding_scheme, args.distance, 
			nystrom=False, parallel=True, **distance_kwargs
		)
		writePickle((node_representations_flatten, dist_matrix), filename=distances_filename)
	
	print(dist_matrix)

	# TODO: integrate and finish all the stats + smoothing matrices...
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
	print()
	print('Prediction indices:')
	print(test_node_idxs)
	print()
	print('Predictions:')
	print(predictions)

	# Evaluate the performance
	error = evaluatePerformance(
		predictions, regression_outputs_flatten[test_node_idxs], normalization='minmax')
	print()
	print(f'Error: {error}')
	
	



if __name__ == '__main__':
	main()
