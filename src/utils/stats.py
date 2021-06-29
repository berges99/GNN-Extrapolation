import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict, Counter

from utils.io import *
from utils.plots import *



# TODO: change performance metrics depending on setting (regression, classification)
def evaluatePerformance(predicted_values, real_values, normalization=None):
	'''
	Function to evaluate the normalized root mean square error (RMSE) of some predictions.
	¡We assume that the predicted and real values are ordered in the same way!

	Parameters:
		- predicted_values: (np.ndarray) Array of graphs with all the predictions for all nodes.
		- real_values: (np.ndarray) Array of graphs with all the real values for all nodes.

	Returns:
		- 
	'''
	# print()
	# print('-' * 30)
	# print('Evaluating performance...')
	assert len(predicted_values) == len(real_values), 'Prediction dimensions do not match!'
	# For all the graphs in the test collection
	n = len(predicted_values)
	rmse = 0
	for i in range(n): #tqdm()
		rmse += (predicted_values[i] - real_values[i])**2
	rmse = np.sqrt(rmse / n)
	# Normalize if specified (NRMSE)
	if normalization:
		if normalization == 'minmax':
			rmse /= max(real_values) - min(real_values)
		elif normalization == 'mean':
			rmse /= np.mean(real_values)
		else:
			print(f'Normalization method ({normalization}) not implemented!')
			return
	return rmse


##########


def getMaxDegree(networkx_dataset):
	'''Auxiliary function that computes the maximum node degree in a networkx dataset.'''
	return np.max([d for sublist in [dict(G.degree).values() for G in networkx_dataset] for d in sublist])


def getAvgDegree(networkx_dataset):
	'''Auxiliary function that computes the average node degree in a networkx dataset.'''
	return np.mean([d for sublist in [dict(G.degree).values() for G in networkx_dataset] for d in sublist])


##########


def computeTreePresence(dataset_rooted_trees_flatten):
	'''Auxiliary function that computes the tree presence distribution in the dataset.'''
	return Counter(dataset_rooted_trees_flatten)


def computeDatasetStats(networkx_dataset,
						dataset_rooted_trees_flatten,
						dist_matrix,
					    filepath, 
					    filename,
					    sample=1):
	'''
	Compute basic dataset stats and stores the result in pretty HTML format.

	Parameters:
		- networkx_dataset: (list<nx.Graph>) List with all the networkx graphs in the dataset.
		- dataset_rooted_trees_flatten: (list<str>) List with all the rooted trees in the dataset.
		- dist_matrix: (np.ndarray) Pairwise distance matrix between all nodes in the dataset.
		- filepath: (str) Full relative path to the dataset.
		- filename: (str) Filename of the dataset.
		- sample: (float) % of graphs in the dataset to use for basic stats computation.

	Returns:
		- None
	'''
	stats_filepath = f"{'/'.join([x for x in filepath.split('/')[:-1]])}/stats"
	###
	# Info from the name
	method, params = parseFilename(filepath, filename)
	N = params['N']
	params = {k: v for k, v in params.items() if k != 'N'}

	###
	# Compute basic networkx stats
	num_graphs_stats = int(sample * len(networkx_dataset))
	graphs_stats = np.random.choice(np.arange(len(networkx_dataset)), size=num_graphs_stats)
	# TBD

	###
	# Tree presence
	tree_counts = computeTreePresence(dataset_rooted_trees_flatten)
	distribution = sorted(list(tree_counts.values()), key=lambda x: -x)
	distribution_filename= f"{stats_filepath}/images/distributions/{filename.rstrip('.pkl')}.png"
	distribution_filename_relative = f"images/distributions/{filename.rstrip('.pkl')}.png"
	plotDistribution(distribution, figname=distribution_filename, title='Pairwise distances.')

	###
	# Plot distance matrix
	heatmap_filename= f"{stats_filepath}/images/heatmaps/{filename.rstrip('.pkl')}.png"
	heatmap_filename_relative = f"images/heatmaps/{filename.rstrip('.pkl')}.png"
	plotHeatmap(dist_matrix, figname=heatmap_filename, title='Distribution of rooted trees.')

	html = f"""<html>
	<body>
	<h1 style="text-align:center;">Dataset Statistics</h1>
	<p style="text-align:center;"><b>Dataset: </b>{filepath}/{filename}</p>

	<ul>
		<li><b>Number of graphs: </b>{N}</li>
		<li><b>Algorithm: </b>{method}</li>
		<li><b>Params: </b>{params}</li>
		<li><b>Tree depth: </b>3</li>
	</ul>

	<h2 style="text-align:center;">Basic stats</h2>

	<h2 style="text-align:center;">Tree presence</h2>
	<p style="text-align:center;">
	<img src={distribution_filename_relative} alt="dist_mat" class="center" height="720" width="1080"></img>
	</p>

	<h2 style="text-align:center;">Pairwise distance matrix</h2>
	<p style="text-align:center;">
	<img src={heatmap_filename_relative} alt="dist_mat" class="center" height="720" width="720"></img>
	</p>

	</body>
	</html>"""

	# Save the HTML
	writeHTML(html, f"{stats_filepath}/{filename.rstrip('.pkl')}.html")
