import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx

from collections import defaultdict

from utils.io import *
from utils.data import *
#from utils.stats import *
from utils.convert import *
from models.dpatterns import *



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--filename', '-f', type=str, default='', help='Filename of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/preferential_attachment', help='Default path to the data.')
    parser.add_argument(
    	'--depth', '-d', type=int, default=3, help='Maximum depth of the rooted trees.')
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


def test(seen_trees, X_test, y_test, agg='mean'):
	''''''
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


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(f'{args.path}/raw')
	# Read the dataset and convert to adjacency lists
	networkx_dataset = readPickle(f'{args.path}/raw/{filename}')
	adj_list_dataset = [getAdjacencyList(G) for G in networkx_dataset]
	# Compute all the d-patterns for every node in the dataset (if necessary)
	if os.path.isfile(f'{args.path}/rooted_trees/{filename}'):
		dataset_rooted_trees = readPickle(f'{args.path}/rooted_trees/{filename}')
	else:
		dataset_rooted_trees = computeDatasetRootedTrees(adj_list_dataset, depth=args.depth)
		writePickle(dataset_rooted_trees, f'{args.path}/rooted_trees/{filename}')
	# Read the teacher outputs of the dataset and split the data
	regression_outputs_filename = getLatestVersion(
		f'{args.path}/teacher_outputs', filename=filename.rstrip('.pkl'))
	regression_outputs = readPickle(f'{args.path}/teacher_outputs/{regression_outputs_filename}')
	X_train, X_test, y_train, y_test = splitData(dataset_rooted_trees, regression_outputs)
	# Check all seen trees in the dataset
	seen_trees = train(X_train, y_train)
	# Use the seen trees to predict on the test data
	predictions = test(seen_trees, X_test, y_test)
	# print(predictions)
	# print(computeDistMatrix(X_train[0]))
	

if __name__ == '__main__':
	main()
