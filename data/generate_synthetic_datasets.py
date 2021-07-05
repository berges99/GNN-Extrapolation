import sys
import time
import random
import argparse
import networkx as nx

from tqdm import tqdm
from collections import defaultdict

sys.path.append('../src')

from utils.io import *



def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    # Train/validation/test splitting
    parser.add_argument(
        '-N', type=int, default=1_000, 
        help='Total number of graphs to generate.')
    parser.add_argument(
        '--test', type=float, default=0.2,
        help='Extra proportion of test graphs to generate for testing.')
    parser.add_argument(
        '--extrapolation', type=float, default=0.2,
        help='Extra proportion of test graphs to generate for extrapolation.')
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


def generateGraphs(N, model, **kwargs):
    '''
    Generate a synthetic dataset with a specific algorithm.
    
    Parameters:
        - N: (int) Number of graphs to generate.
        - model: (str) Algorithm for generating the data.
        (**kwargs) Extra keyword arguments depending on the chosen model.

    Returns:
        - (list<nx.Graph>) List with the generated graphs.

    '''
    print()
    print('Generating synthetic data...')
    print('-' * 30)
    dataset = defaultdict(list)
    # Generate N total amount of graphs
    for i in tqdm(range(N)):
        if model == 'erdos_renyi':
            dataset['train'].append(nx.erdos_renyi_graph(
                n=uniformSample(kwargs['n']), p=uniformSample(kwargs['p'])))
        elif model == 'preferential_attachment':
            dataset['train'].append(nx.barabasi_albert_graph(
                n=uniformSample(kwargs['n']), m=uniformSample(kwargs['m'])))
    # Generate approximately 20% of test data
    for i in tqdm(range(int(0.2 * N))):
        if model == 'erdos_renyi':
            dataset['test'].append(nx.erdos_renyi_graph(
                n=uniformSample(kwargs['n']), p=uniformSample(kwargs['p'])))
        elif model == 'preferential_attachment':
            dataset['test'].append(nx.barabasi_albert_graph(
                n=uniformSample(kwargs['n']), m=uniformSample(kwargs['m'])))
    # Generate approximately 20% of extrapolation data
    for i in tqdm(range(int(0.2 * N))):
        if model == 'erdos_renyi':
            dataset['extrapolation'].append(nx.erdos_renyi_graph(
                n=uniformSample((1.5 * np.array(kwargs['n'])).astype(int)), p=uniformSample(kwargs['p'])))
        elif model == 'preferential_attachment':
            dataset['extrapolation'].append(nx.barabasi_albert_graph(
                n=uniformSample((1.5 * np.array(kwargs['n'])).astype(int)), m=uniformSample(kwargs['m'])))
    return dataset


def main():
    args = readArguments()
    if args.model == 'erdos_renyi':
        dataset = generateGraphs(args.N, args.model, n=args.n, p=args.p)
        filename = f"synthetic/{args.model}/N{args.N}_n{'-'.join(map(str, args.n))}" \
                   f"_p{'-'.join(map(str, args.p))}_{int(time.time())}/raw_networkx.pkl"
    else: #elif args.model == 'preferential_attachment':
        dataset = generateGraphs(args.N, args.model, n=args.n, m=args.m)
        filename = f"synthetic/{args.model}/N{args.N}_n{'-'.join(map(str, args.n))}" \
                   f"_m{'-'.join(map(str, args.m))}_{int(time.time())}/raw_networkx.pkl"
    # Store the dataset into memory
    writePickle(dataset, filename=filename)
    

if __name__ == '__main__':
    main()
