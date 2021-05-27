import numpy as np
import matplotlib.pyplot as plt

from functools import wraps



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
				plt.savefig(f"{kwargs['figname']}.png")
			# Display all "open" (non-closed) figures if necessary
			if display:
				plt.show()
		return wrapper
	return inner


@displayPlot
def plotHeatmap(matrix, figname, title=''):
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.imshow(matrix)
	ax.set_title('Pairwise distances')
	ax.plot([x for x in range(10)], [x for x in range(10)])
	return fig, figname


@displayPlot
def plotHeatmap(matrix, figname):
	fig, ax = plt.subplots(figsize=(12, 12))
	ax.imshow(matrix)
	ax.set_title('Pairwise distances')
	ax.plot([x for x in range(10)], [x for x in range(10)])
	return fig, figname