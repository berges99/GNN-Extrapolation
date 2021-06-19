import os
import pickle
import argparse
import numpy as np

from pathlib import Path
from functools import wraps



##########


class KeepOrderAction(argparse.Action):
	'''Aux class to keep the order of the arguments that are given through the terminal.'''
	def __call__(self, parser, namespace, values, option_string=None):
		if not 'ordered_args' in namespace:
			setattr(namespace, 'ordered_args', [])
		previous = namespace.ordered_args
		previous.append((self.dest, values))
		setattr(namespace, 'ordered_args', previous)
		# Maintain the values in the regular argparse labels
		setattr(namespace, self.dest, values)


def parameterizedKeepOrderAction(additional_arg, keep=True):
	'''Aux wrapper to customize the name of the saved ordered arguments in the special KeepOrderAction class.'''
	class KPA(argparse.Action):
		'''Aux class to keep the order of the arguments that are given through the terminal.'''
		def __call__(self, parser, namespace, values, option_string=None):
			if not additional_arg in namespace:
				setattr(namespace, additional_arg, [])
			previous = getattr(namespace, additional_arg)
			previous.append((self.dest, values))
			setattr(namespace, additional_arg, previous)
			# Maintain the values in the regular argparse labels
			if keep:
				setattr(namespace, self.dest, values)
	return KPA


##########


def buildPath(f):
	'''Auxiliary wrapper that explicitly builds the parent path of a file.'''
	@wraps(f)
	def wrapper(*args, **kwargs):
		filepath = '/'.join(kwargs['filename'].split('/')[:-1])
		Path(filepath).mkdir(parents=True, exist_ok=True)
		return f(*args, **kwargs)
	return wrapper


@buildPath
def writeHTML(body, filename):
	'''Auxiliary function that writes an HTML file into memory.'''
	with open(filename, 'w') as f:
		f.write(body)


@buildPath
def writeNumpy(arrays, filename):
	'''Auxiliary function that writes an np file into memory.'''
	with open(filename, 'wb') as f:
		f.write(arrays)


@buildPath
def writePickle(dataset, filename):
	'''Auxiliary function that writes a list of networkx graphs as serialized python objects.'''
	with open(filename, 'wb') as f:
		pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def readPickle(filename):
	'''Auxiliary function that writes a list of networkx graphs as serialized python objects.'''
	with open(filename, 'rb') as f:
		return pickle.load(f)


def getLatestVersion(filepath, start='', end=''):
	'''Auxiliary function that returns the latest created version at a given path.'''
	versions = [version for version in os.listdir(filepath) if version.endswith(end)]
	if start:
		versions = [version for version in versions if version.startswith(start)]
	ts = np.array([int(version.rstrip(end).split('_')[-1]) for version in versions])
	return versions[np.argmax(ts)]


def parseFilename(filepath, filename, extension='.pkl'):
	'''Auxiliary function that parses the name of a given file and returns the important information.'''
	filepath = filepath.split('/')[:-1]
	if filepath[-2] == 'synthetic':
		filename = filename.rstrip(extension).split('_')[:-1]
		# Return the model and all the parameters of the model
		return filepath[-1], {param[0]: param[1:].split('-') for param in filename}
	else:
		return filepath[-1], None
