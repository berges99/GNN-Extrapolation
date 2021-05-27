import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.io import *
from utils.plots import *



def evaluatePerformance(predicted_values, real_values):
	'''
	Function to evaluate the performance of some predictions.
	¡We assume that the predicted and real values are ordered in the same way!

	Parameters:
		- predicted_values: (np.ndarray) Array of graphs with all the predictions for all nodes.
		- real_values: (np.ndarray) Array of graphs with all the real values for all nodes.

	Returns:
		- 
	'''
	print()
	print('-' * 30)
	print('Evaluating performance...')
	assert len(predicted_values) == len(real_values), 'Prediction dimensions do not match!'
	# For all the graphs in the test collection
	total_error = 0
	for i, n in enumerate(tqdm(real_values)):
		total_error += np.abs(predicted_values[i] - real_values[i])
	avg_G_error = total_error / (len(predicted_values) / 30)
	avg_n_error = avg_G_error / 30
	return total_error, avg_G_error, avg_n_error


def plotPresenceDistribution(filename=None):
	'''
	Plots the distribution of appearances of seen trees in the given dataset.

	Parameters:
		- 

	Returns:
		- (str) Filepath of the resulting figure.
	'''
	pass


def computeStats(seen_trees, filepath, filename, unseen_trees=None):
	'''
	Compute simple dataset stats (mainly trees presence and counts).

	Parameters:
		- seen_trees: (dict<str, list>) Dictionary with the seen trees.
		- filename: (str) String with the path where to store the stats (None otherwise).
		- unseen_trees: (dict<str, list>) Dictionary with the unseen/test trees (None otherwise).

	Returns:
		- (str) HTML with the parsed results.
	'''
	method, params = parseFilename(filepath, filename)
	N = params['N']
	params = {k: v for k, v in params.items() if k != 'N'}

	html = f"""<html>
	<body>
	<h1 style="text-align:center;">Dataset Statistics</h1>
	<p style="text-align:center;"><b>Dataset: </b>{filepath + filename}</p>

	<ul>
		<li><b>Number of graphs: </b>{N}</li>
		<li><b>Algorithm: </b>{method}</li>
		<li><b>Params: </b>{params}</li>
		<li><b>Tree depth: </b>3</li>
	</ul>

	</body>
	</html>"""

	# Save the HTML
	writeHTML(html, f"{'/'.join([x for x in filepath.split('/')[:-1]])}/stats/{filename}.html")
