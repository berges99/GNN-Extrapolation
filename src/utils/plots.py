import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from mpl_toolkits.axes_grid1 import make_axes_locatable



def displayPlot(display=False, save=True):
	'''
	Plot wrapper to specify if the figure should be displayed and/or saved.

	Parameters:
		- display: (bool) Whether to interactively display the figure or not.
		- save: (bool) Whether to save the figure or not.

	Returns:
		- None
	'''
	def inner(f):
		@wraps(f)
		def wrapper(*args, **kwargs):
			# Turn interactive plotting off
			plt.ioff()
			# Generate the plot
			fig = f(*args, **kwargs)
			# Save figure if necessary
			if save:
				plt.savefig(kwargs['figname'])
			# Display all "open" (non-closed) figures if necessary
			if display:
				plt.show()
		return wrapper
	return inner


@displayPlot()
def plotHeatmap(matrix, figname, title=''):
	'''Auxiliary function to plot and save a heatmap.'''
	fig, ax = plt.subplots(figsize=(16, 16))
	heatmap = ax.imshow(matrix)
	# Create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	fig.colorbar(heatmap, cax=cax)
	ax.set_title(title)
	return fig


@displayPlot()
def plotDistribution(distribution, figname, limit=1_000, title=''):
	'''Auxiliary function to plot and save a distribution.'''
	limit = min(limit, len(distribution))
	fig, ax = plt.subplots(figsize=(20, 16))
	ax.plot(np.arange(limit), distribution[:limit], c='indianred', lw=3)
	ax.set_ylabel('Counts')
	ax.set_xlabel('Idxs')
	ax.set_title(title)
	return fig
