import os
import torch
import argparse
import importlib
import numpy as np
import networkx as nx

from tqdm import tqdm

from torch import nn
from torch_geometric.data import DataLoader

from utils.io import *
from utils.data import *
from utils.stats import *
from utils.convert import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
    	'--filename', '-f', type=str, default='', help='Filename of the dataset to be used.')
    parser.add_argument(
    	'--path', '-p', type=str, default='../data/synthetic/preferential_attachment', help='Default path to the data.')
    # parser.add_argument(
    # 	'--model', '-m', type=str, default='GIN', help='Graph Neural Network architecture to be used.')
    parser.add_argument(
    	'--setting', '-s', type=str, default='continuous', help='Setting used for producing outputs [continuous, categorical].')
    return parser.parse_args()


def runWL(model, loader, device):
	'''


	TBD

	'''
	#model.eval()
	node_embeddings = []
	for data in tqdm(loader):
		data = data.to(device)
		output = model(data)
		node_embeddings.append(output.detach().numpy())
	return node_embeddings


def main():
	args = readArguments()
	filename = args.filename or getLatestVersion(f'{args.path}/raw')
	# Read the dataset and convert it to torch_geometric.data
	networkx_dataset = readPickle(f'{args.path}/raw/{filename}')
	torch_dataset = fromNetworkx2Torch(networkx_dataset, add_degree=True)
	data_loader = DataLoader(torch_dataset, batch_size=1)


	# Import the model
	module = importlib.import_module(f'models.WL.{args.setting}')
	# Init the model
	model = module.WL(in_channels=1).to(device)

	# Run the model and generate the WL embeddings
	node_embeddings = runWL(model, data_loader, device)

	print(len(node_embeddings))
	print(node_embeddings[0].shape)

	

	
	# ##########
	# # Compute all the d-patterns for every node in the dataset (if necessary)
	# if os.path.isfile(f'{args.path}/rooted_trees/{filename}'):
	# 	dataset_rooted_trees = readPickle(f'{args.path}/rooted_trees/{filename}')
	# else:
	# 	dataset_rooted_trees = computeDatasetRootedTrees(
	# 		adj_list_dataset, method=args.method, depth=args.depth, parallel=False)
	# 	writePickle(dataset_rooted_trees, f'{args.path}/rooted_trees/{filename}')
	
	# ##########
	# # Compute the pairwise distance matrix if necessary (relabel if indicated)
	# if os.path.isfile(f'{args.path}/dist_matrices/{filename}'):
	# 	dataset_rooted_trees_flatten, dist_matrix = readPickle(f'{args.path}/dist_matrices/{filename}')
	# else:
	# 	dataset_rooted_trees_flatten = [
	# 		re.sub(r"\d+", 'x', item) if args.relabel else item
	# 		for sublist in dataset_rooted_trees for item in sublist
	# 	]
	# 	dist_matrix = computeDistMatrix(
	# 		dataset_rooted_trees_flatten, method=args.method, nystrom=False, parallel=True)
	# 	writePickle((dataset_rooted_trees_flatten, dist_matrix), f'{args.path}/dist_matrices/{filename}')
	# ##########
	# # Compute and store basic dataset stats
	# computeDatasetStats(networkx_dataset, dataset_rooted_trees_flatten, dist_matrix, filepath=f'{args.path}/rooted_trees', filename=filename, sample=1)
	# ##########
	# # Read the teacher outputs of the dataset and split the data
	# regression_outputs_filename = getLatestVersion(
	# 	f'{args.path}/teacher_outputs', filename=filename.rstrip('.pkl'))
	# regression_outputs = readPickle(f'{args.path}/teacher_outputs/{regression_outputs_filename}')
	# regression_outputs_flatten = np.array([item for sublist in regression_outputs for item in sublist])
	# # Generate random shuffle
	# train_idxs, test_idxs = generateShuffle(len(dataset_rooted_trees), sort=True)
	# ##########
	# # Predict on test data with the baseline method
	# test_node_idxs, predictions = baseline(
	# 	dataset_rooted_trees_flatten, regression_outputs_flatten, dist_matrix, train_idxs, test_idxs, smoothed=True)
	# print(test_node_idxs, predictions)
	# # Evaluate the performance
	# total_error, avg_G_error, avg_n_error = evaluatePerformance(predictions, regression_outputs_flatten[test_node_idxs])
	# print(total_error, avg_G_error, avg_n_error)
	
	



	
		
	

if __name__ == '__main__':
	main()
