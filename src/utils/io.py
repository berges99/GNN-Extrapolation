import os
import pickle
import numpy as np



def writeHTML(body, filename):
	'''Auxiliary function that writes an HTML file into memory.'''
	with open(filename, 'w') as f:
		f.write(body)


def writeNumpy(arrays, filename):
	'''Auxiliary function that writes an np file into memory.'''
	with open(filename, 'wb') as f:
		f.write(arrays)


def writePickle(dataset, filename):
	'''Auxiliary function that writes a list of networkx graphs as serialized python objects.'''
	with open(filename, 'wb') as f:
		pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def readPickle(filename):
	'''Auxiliary function that writes a list of networkx graphs as serialized python objects.'''
	with open(filename, 'rb') as f:
		return pickle.load(f)


def getLatestVersion(filepath, filename=None, extension='.pkl'):
	'''Auxiliary function that returns the latest created version at a given path.'''
	versions = [version for version in os.listdir(filepath) if version.endswith(extension)]
	if filename:
		versions = [version for version in versions if version.startswith(filename)]
	ts = np.array([int(version.rstrip(extension).split('_')[-1]) for version in versions])
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
