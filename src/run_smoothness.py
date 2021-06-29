import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

from numba import jit
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from utils.smoothness import *
from utils.stats import evaluatePerformance
from utils.convert import expandCompressedDistMatrix
from utils.io import readPickle, writePickle, booleanString



# Available implemented methods (fetch all implemented methods in the models subfolder)
IMPLEMENTED_MODELS = [
    m.rstrip('.py') for m in os.listdir('models')
    if os.path.isfile(f'models/{m}') and m[0].isupper() and m.endswith('.py')
]


def readArguments():
    '''Auxiliary function to parse the arguments passed to the script.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dist_matrix_filename', type=str, required=True,
        help='Full relative path to the node representations to be used by the baseline method.')
    parser.add_argument(
        '--teacher_outputs_filename', type=str, required=True, 
        help='Full relative path to the teacher outputs of the given dataset (or parent directory path).')
    parser.add_argument(
        '--verbose', '-v', type=booleanString, default=False, 
        help='Whether to print the outputs through the terminal.')
    return parser.parse_args()


def resolveTeacherOutputsFilenames(path):
    '''Auxiliary function that returns an iterable with all the involved teacher outputs.'''
    if os.path.isdir(path):
        teacher_outputs_filenames = [
            f"{path}/{t_i}/teacher_outputs.pkl" for t_i in os.listdir(path)
            if all(c.isdigit() for c in t_i)
        ]
    else:
        teacher_outputs_filenames = [path]
    return teacher_outputs_filenames


def main():
    args = readArguments()
    # Read the dist matrix and node representations for the specified data
    node_representations, dist_matrix = readPickle(args.dist_matrix_filename)
    # Expand the compressed distance matrix
    dist_matrix = expandCompressedDistMatrix(dist_matrix)


    # print(np.unique(dist_matrix, return_counts=True))
    # deciles = np.percentile(dist_matrix, np.arange(0, 101, 1))
    # print(deciles)
    # return


    # Compute the graph laplacian
    print()
    print('-' * 30)
    print('Computing W...')
    W = computeWNumba(dist_matrix)
    print()
    print('-' * 30)
    print('Computing combinatorial Laplacian...')
    L = computeLaplacian(W, normalize=False)
    # Read the teacher outputs of the dataset (iterate through folder if necessary)
    print()
    print('-' * 30)
    print('Iterating through different teacher outputs...')
    for teacher_outputs_filename in tqdm(resolveTeacherOutputsFilenames(args.teacher_outputs_filename)):
        # Read the teacher outputs
        teacher_outputs = readPickle(teacher_outputs_filename)
        teacher_outputs_flatten = [item for sublist in teacher_outputs for item in sublist]
        # Student smoothness will have the shape:
        #     {'model_i': {'model_config_i': {'epoch_i': {'stat_i': [...]}}}}
        student_smoothness = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        # For every model used for generating outputs
        for model in IMPLEMENTED_MODELS:
            student_outputs_filename_prefix = \
                f"{'/'.join(teacher_outputs_filename.split('/')[:-1])}/student_outputs/{model}"
            if os.path.isdir(student_outputs_filename_prefix):
                # Baseline outputs have a specific folder structure
                if model == 'Baseline':
                    # Get all student outputs_filenames
                    student_outputs_filenames = [
                        f"{student_outputs_filename_prefix}/{x}" for x in os.listdir(student_outputs_filename_prefix)
                        if x.endswith('.pkl')
                    ]
                    for student_outputs_filename in student_outputs_filenames:
                        model_config = student_outputs_filename.split('/')[-1].rstrip('.pkl')
                        student_output = readPickle(student_outputs_filename)[0] # It only has one realization
                        if len(np.array(student_output).shape) == 2:
                            student_output = [item for sublist in student_output for item in sublist]
                        # Compute the root mean square error of the predictions
                        rmse = evaluatePerformance(student_output, teacher_outputs_flatten, normalization='minmax')
                        student_smoothness[model][model_config][0]['rmse'].append(rmse)
                        # Compute the smoothness of such predictions
                        student_output = np.array(student_output)[:, None]
                        smoothness = computeSmoothness(L, F=student_output)
                        student_smoothness[model][model_config][0]['smoothness'].append(smoothness)
                        student_smoothness[model][model_config][0] = {
                            'mean_smoothness': student_smoothness[model][model_config][0]['smoothness'][0],
                            'std_dev_smoothness': 0.0,
                            'mean_rmse': student_smoothness[model][model_config][0]['rmse'][0],
                            'std_dev_rmse': 0.0,
                        }
                        # Prepare data for pandas multiindex columns
                        student_smoothness[model][model_config] = {
                            (outer_key, inner_key): values
                            for outer_key, inner_dict in student_smoothness[model][model_config].items()
                            for inner_key, values in inner_dict.items()
                        }
                # Any other implemented model
                else:
                    # Get all student parameterizations
                    student_parameterizations = [
                        p for p in os.listdir(student_outputs_filename_prefix) if 'epochs' in p
                    ]
                    for student_parameterization in student_parameterizations:
                        model_config = student_parameterization.split('__epochs')[0]
                        # Get all available student outputs for the given parameterization
                        student_outputs_filenames = [
                            f"{student_outputs_filename_prefix}/{student_parameterization}/{ts}/student_outputs.pkl"
                            for ts in os.listdir(f"{student_outputs_filename_prefix}/{student_parameterization}")
                            if all(c.isdigit() for c in ts)
                        ]
                        for student_outputs_filename in student_outputs_filenames:
                            student_outputs = readPickle(student_outputs_filename)
                            # Iterate through all the student outputs (if epochs > 1)
                            for epoch, student_output in enumerate(student_outputs):
                                if len(np.array(student_output).shape) == 2:
                                    student_output = [item for sublist in student_output for item in sublist]
                                # Compute the root mean square error of the predictions
                                rmse = evaluatePerformance(student_output, teacher_outputs_flatten, normalization='minmax')
                                student_smoothness[model][model_config][epoch]['rmse'].append(rmse)
                                # Compute the smoothness of such predictions
                                student_output = np.array(student_output)[:, None]
                                smoothness = computeSmoothness(L, F=student_output)
                                student_smoothness[model][model_config][epoch]['smoothness'].append(smoothness)
                        # Compute mean and std deviation for all initializations (per epoch approach)
                        for epoch, stats in student_smoothness[model][model_config].items():
                            student_smoothness[model][model_config][epoch] = {
                                'mean_smoothness': np.mean(stats['smoothness']),
                                'std_dev_smoothness': np.std(stats['smoothness']),
                                'mean_rmse': np.mean(stats['rmse']),
                                'std_dev_rmse': np.std(stats['rmse']),
                            }
                        # Prepare data for pandas multiindex columns
                        student_smoothness[model][model_config] = {
                            (outer_key, inner_key): values
                            for outer_key, inner_dict in student_smoothness[model][model_config].items()
                            for inner_key, values in inner_dict.items()
                        }
        # Prepare data for pandas multiindex index
        student_smoothness = {
            (outer_key, inner_key): values
            for outer_key, inner_dict in student_smoothness.items()
            for inner_key, values in inner_dict.items()
        }
        # Convert into pandas dataframe
        student_smoothness = pd.DataFrame.from_dict(student_smoothness, orient='index')
        if args.verbose:
            print()
            print('-' * 30)
            print(f'Smoothness for {teacher_outputs_filename}:')
            print(student_smoothness)
        # Store file
        student_smoothness_filename = \
            f"{'/'.join(teacher_outputs_filename.split('/')[:-1])}/smoothness/" \
            f"{('/'.join(args.dist_matrix_filename.split('/')[-6:-3]) + '/' + '/'.join(args.dist_matrix_filename.split('/')[-2:])).replace('/', '__').rstrip('.pkl')}.csv"
        student_smoothness_filepath = '/'.join(student_smoothness_filename.split('/')[:-1])
        Path(student_smoothness_filepath).mkdir(parents=True, exist_ok=True)
        student_smoothness.to_csv(student_smoothness_filename)
                

if __name__ == '__main__':
    main()
