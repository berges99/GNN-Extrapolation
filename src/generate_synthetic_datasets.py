import time
import random
import argparse
import networkx as nx

from tqdm import tqdm

from utils.io import *



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    # Train/validation/test splitting
    parser.add_argument(
    	'-N', type=int, default=1_000, help='')
    # Add parameters for the specific random graph generation algorithms
    subparsers = parser.add_subparsers(dest='model')
    # Parse the specific erdos_renyi model arguments
    erdos_renyi = subparsers.add_parser('erdos_renyi', help='Erdos-Renyi model parser.')
    erdos_renyi.add_argument(
    	'-n', required=True, type=int, nargs='+', help='')
    erdos_renyi.add_argument(
    	'-p', required=True, type=float, nargs='+', help='')
    # Preferential attachment model arguments
    preferential_attachment = subparsers.add_parser('preferential_attachment', help='PA model parser.')
    preferential_attachment.add_argument(
    	'-n', required=True, type=int, nargs='+', help='')
    preferential_attachment.add_argument(
    	'-m', required=True, type=int, nargs='+', help='')
    return parser.parse_args()


def uniformSample(values):
	'''Auxiliary function that samples uniform values between two thresholds.'''
	if len(values) == 2:
		if isinstance(values[0], int):
			return random.randint(values[0], values[1])
		else:
			return random.uniform(values[0], values[1])
	else:
		return values[0]


def generateGraphs(N, model, *args, **kwargs):
	'''
	Generate a synthetic dataset with a specific algorithm.
	
	Parameters:
		- N: (int) Number of graphs to generate.
		- model: (str) Algorithm for generating the data.
		- (...) extra parameters depending on the chosen model.

	Returns:
		- (list<nx.Graph>) List with the generated graphs.

	'''
	print()
	print('-' * 30)
	print('Generating synthetic data...')
	dataset = []
	# Generate N total amount of graphs
	for i in tqdm(range(N)):
		# Generate erdos renyi graphs
		if model == 'erdos_renyi':
			dataset.append(nx.erdos_renyi_graph(
				n=uniformSample(kwargs['n']), p=uniformSample(kwargs['p'])))
		elif model == 'preferential_attachment':
			dataset.append(nx.barabasi_albert_graph(
				n=uniformSample(kwargs['n']), m=uniformSample(kwargs['m'])))
	return dataset


def main():
	args = readArguments()
	if args.model == 'erdos_renyi':
		dataset = generateGraphs(args.N, args.model, n=args.n, p=args.p)
		filename = f"../data/synthetic/{args.model}/raw/n{'-'.join(map(str, args.n))}" \
				   f"_p{'-'.join(map(str, args.p))}_{int(time.time())}.pkl"
	else: #elif args.model == 'preferential_attachment':
		dataset = generateGraphs(args.N, args.model, n=args.n, m=args.m)
		filename = f"../data/synthetic/{args.model}/raw/n{'-'.join(map(str, args.n))}" \
				   f"_m{'-'.join(map(str, args.m))}_{int(time.time())}.pkl"
	# Store the dataset into memory
	writePickle(dataset, filename)
	

if __name__ == '__main__':
	main()
