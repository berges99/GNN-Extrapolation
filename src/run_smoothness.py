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
        '--verbose', type=booleanString, default=False, 
        help='Whether to print the outputs through the terminal.')
    return parser.parse_args()


def resolveTeacherOutputsFilenames(path):
    '''Auxiliary function that returns an iterable with all the involved teacher outputs.'''
    if 'teacher_outputs.pkl' in os.listdir(path):
        teacher_outputs_filenames = [path]
    else:
        teacher_outputs_filenames = [
            f"{path}/{t_i}/teacher_outputs.pkl" for t_i in os.listdir(path)
            if all(c.isdigit() for c in t_i[-13:])
        ]
    return teacher_outputs_filenames


def main():
    args = readArguments()
    # Read the distance matrices. Example of path:
    # ../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sMaxdegree/train/full64.npz
    dist_matrix_train = np.load(args.dist_matrix_filename)
    idxs = dist_matrix_train['idxs'] if 'nystrom' in args.dist_matrix_filename.split('/')[-1] else None
    dist_matrix_train = dist_matrix_train['dist_matrix']
    # Read the test and extrapolation matrices
    dist_matrix_test = np.load(
        f"{'/'.join(args.dist_matrix_filename.split('/')[:-2])}/test/full64.npz")['dist_matrix']
    dist_matrix_extrapolation = np.load(
        f"{'/'.join(args.dist_matrix_filename.split('/')[:-2])}/extrapolation/full64.npz")['dist_matrix']
    # Compute the induced graph adjacency matrix (train, test, extrapolation)
    print()
    print('Computing Ws...')
    print('-' * 30)
    W_train = computeW(dist_matrix_train, threshold=True, normalization=True)
    W_test = computeW(dist_matrix_test, threshold=True, normalization=True)
    W_extrapolation = computeW(dist_matrix_extrapolation, threshold=True, normalization=True)
    # Compute the diagonal matrix and the graph combinatorial Laplacian (train, test, extrapolation)
    print()
    print('Computing combinatorial Laplacians...')
    print('-' * 30)
    D_train, L_train = computeL(W_train, idxs=idxs, normalize=False)
    D_test, L_test = computeL(W_test, idxs=None, normalize=False)
    D_extrapolation, L_extrapolation = computeL(W_extrapolation, idxs=None, normalize=False)
    # Read the teacher outputs of the dataset (iterate through folder if necessary)
    print()
    print('Iterating through different teacher outputs...')
    print('-' * 30)
    for teacher_outputs_filename in tqdm(resolveTeacherOutputsFilenames(args.teacher_outputs_filename)):
        # Read the teacher outputs (train, test, extrapolation)
        teacher_outputs_train = readPickle(teacher_outputs_filename)['train']
        teacher_outputs_train_flatten = [item for sublist in teacher_outputs_train for item in sublist]
        teacher_outputs_test = readPickle(teacher_outputs_filename)['test']
        teacher_outputs_test_flatten = [item for sublist in teacher_outputs_test for item in sublist]
        teacher_outputs_extrapolation = readPickle(teacher_outputs_filename)['extrapolation']
        teacher_outputs_extrapolation_flatten = [item for sublist in teacher_outputs_extrapolation for item in sublist]
        # Student smoothness will have the shape:
        #     {'model_i': {'model_config_i': {'epoch_i': {'stat_i': [...]}}}}
        student_smoothness = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
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
                        # Read student outputs (they have no epochs [0])
                        student_output_train = readPickle(student_outputs_filename)['train'][0]
                        student_output_test = readPickle(student_outputs_filename)['test'][0]
                        student_output_extrapolation = readPickle(student_outputs_filename)['extrapolation'][0]
                        if len(np.array(student_output_train).shape) == 2:
                            student_output_train = [item for sublist in student_output_train for item in sublist]
                            student_output_test = [item for sublist in student_output_test for item in sublist]
                            student_output_extrapolation = [item for sublist in student_output_extrapolation for item in sublist]
                        # Compute the root mean square error of the predictions (train, test, extrapolation)
                        rmse_train = evaluatePerformance(
                            student_output_train, teacher_outputs_train_flatten, normalization='minmax')
                        rmse_test = evaluatePerformance(
                            student_output_test, teacher_outputs_test_flatten, normalization='minmax')
                        rmse_extrapolation = evaluatePerformance(
                            student_output_extrapolation, teacher_outputs_extrapolation_flatten, normalization='minmax')
                        student_smoothness[model][model_config][0]['rmse']['train'].append(rmse_train)
                        student_smoothness[model][model_config][0]['rmse']['test'].append(rmse_test)
                        student_smoothness[model][model_config][0]['rmse']['extrapolation'].append(rmse_extrapolation)
                        # Compute the smoothness of such predictions
                        student_output_train = np.array(student_output_train)[:, None]
                        student_output_test = np.array(student_output_test)[:, None]
                        student_output_extrapolation = np.array(student_output_extrapolation)[:, None]
                        smoothness_train = computeSmoothness(D_train, L_train, f=student_output_train)
                        smoothness_test = computeSmoothness(D_test, L_test, f=student_output_test)
                        smoothness_extrapolation = computeSmoothness(D_extrapolation, L_extrapolation, f=student_output_extrapolation)
                        student_smoothness[model][model_config][0]['smoothness']['train'].append(smoothness_train)
                        student_smoothness[model][model_config][0]['smoothness']['test'].append(smoothness_test)
                        student_smoothness[model][model_config][0]['smoothness']['extrapolation'].append(smoothness_extrapolation)
                        student_smoothness[model][model_config][0] = {
                            'train': {
                                'mean_smoothness': student_smoothness[model][model_config][0]['smoothness']['train'][0],
                                'std_dev_smoothness': 0.0,
                                'mean_rmse': student_smoothness[model][model_config][0]['rmse']['train'][0],
                                'std_dev_rmse': 0.0,
                            },
                            'test': {
                                'mean_smoothness': student_smoothness[model][model_config][0]['smoothness']['test'][0],
                                'std_dev_smoothness': 0.0,
                                'mean_rmse': student_smoothness[model][model_config][0]['rmse']['test'][0],
                                'std_dev_rmse': 0.0,
                            },
                            'extrapolation': {
                                'mean_smoothness': student_smoothness[model][model_config][0]['smoothness']['extrapolation'][0],
                                'std_dev_smoothness': 0.0,
                                'mean_rmse': student_smoothness[model][model_config][0]['rmse']['extrapolation'][0],
                                'std_dev_rmse': 0.0,
                            },
                        }
                        # Prepare data for pandas multiindex columns
                        student_smoothness[model][model_config] = {
                            (epoch, setting, kpi): values
                            for epoch, inner_dict in student_smoothness[model][model_config].items()
                            for setting, inner_dict2 in inner_dict.items()
                            for kpi, values in inner_dict2.items()
                        }
                # Any other implemented model
                else:
                    # Get all student parameterizations
                    student_parameterizations = [
                        p for p in os.listdir(student_outputs_filename_prefix) if 'init' in p
                    ]
                    for student_parameterization in student_parameterizations:
                        model_config = student_parameterization #.split('__epochs')[0]
                        # Get all available student outputs for the given parameterization
                        student_outputs_filenames = [
                            f"{student_outputs_filename_prefix}/{student_parameterization}/{ts}/student_outputs.pkl"
                            for ts in os.listdir(f"{student_outputs_filename_prefix}/{student_parameterization}")
                            if all(c.isdigit() for c in ts[-13:])
                        ]
                        for student_outputs_filename in student_outputs_filenames:
                            student_outputs_train = readPickle(student_outputs_filename)['train']
                            student_outputs_test = readPickle(student_outputs_filename)['test']
                            student_outputs_extrapolation = readPickle(student_outputs_filename)['extrapolation']
                            # Iterate through all the student outputs (if epochs > 1)
                            for epoch, (student_output_train, student_output_test, student_output_extrapolation) in enumerate(zip(student_outputs_train, student_outputs_test, student_outputs_extrapolation)):
                                if len(np.array(student_output_train).shape) == 2:
                                    student_output_train = [item for sublist in student_output_train for item in sublist]
                                    student_output_test = [item for sublist in student_output_test for item in sublist]
                                    student_output_extrapolation = [item for sublist in student_output_extrapolation for item in sublist]
                                # Compute the root mean square error of the predictions (train, test, extrapolation)
                                rmse_train = evaluatePerformance(
                                    student_output_train, teacher_outputs_train_flatten, normalization='minmax')
                                rmse_test = evaluatePerformance(
                                    student_output_test, teacher_outputs_test_flatten, normalization='minmax')
                                rmse_extrapolation = evaluatePerformance(
                                    student_output_extrapolation, teacher_outputs_extrapolation_flatten, normalization='minmax')
                                student_smoothness[model][model_config][epoch]['rmse']['train'].append(rmse_train)
                                student_smoothness[model][model_config][epoch]['rmse']['test'].append(rmse_test)
                                student_smoothness[model][model_config][epoch]['rmse']['extrapolation'].append(rmse_extrapolation)
                                # Compute the smoothness of such predictions
                                student_output_train = np.array(student_output_train)[:, None]
                                student_output_test = np.array(student_output_test)[:, None]
                                student_output_extrapolation = np.array(student_output_extrapolation)[:, None]
                                smoothness_train = computeSmoothness(D_train, L_train, f=student_output_train)
                                smoothness_test = computeSmoothness(D_test, L_test, f=student_output_test)
                                smoothness_extrapolation = computeSmoothness(D_extrapolation, L_extrapolation, f=student_output_extrapolation)
                                student_smoothness[model][model_config][epoch]['smoothness']['train'].append(smoothness_train)
                                student_smoothness[model][model_config][epoch]['smoothness']['test'].append(smoothness_test)
                                student_smoothness[model][model_config][epoch]['smoothness']['extrapolation'].append(smoothness_extrapolation)
                        # Compute mean and std deviation for all initializations (per epoch approach)
                        for epoch, stats in student_smoothness[model][model_config].items():
                            student_smoothness[model][model_config][epoch] = {
                                'train': {
                                    'mean_smoothness': np.mean(stats['smoothness']['train']),
                                    'std_dev_smoothness': np.std(stats['smoothness']['train']),
                                    'mean_rmse': np.mean(stats['rmse']['train']),
                                    'std_dev_rmse': np.std(stats['rmse']['train']),
                                },
                                'test': {
                                    'mean_smoothness': np.mean(stats['smoothness']['test']),
                                    'std_dev_smoothness': np.std(stats['smoothness']['test']),
                                    'mean_rmse': np.mean(stats['rmse']['test']),
                                    'std_dev_rmse': np.std(stats['rmse']['test']),
                                },
                                'extrapolation': {
                                    'mean_smoothness': np.mean(stats['smoothness']['extrapolation']),
                                    'std_dev_smoothness': np.std(stats['smoothness']['extrapolation']),
                                    'mean_rmse': np.mean(stats['rmse']['extrapolation']),
                                    'std_dev_rmse': np.std(stats['rmse']['extrapolation']),
                                }
                            }
                        # Prepare data for pandas multiindex columns
                        student_smoothness[model][model_config] = {
                            (epoch, setting, kpi): values
                            for epoch, inner_dict in student_smoothness[model][model_config].items()
                            for setting, inner_dict2 in inner_dict.items()
                            for kpi, values in inner_dict2.items()
                        }
        # Prepare data for pandas multiindex index
        student_smoothness = {
            (model, model_config): values
            for model, inner_dict in student_smoothness.items()
            for model_config, values in inner_dict.items()
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
            f"{'/'.join(args.dist_matrix_filename.split('/')[-8:]).replace('/', '__').rstrip('.npz')}.csv"
        student_smoothness_filepath = '/'.join(student_smoothness_filename.split('/')[:-1])
        Path(student_smoothness_filepath).mkdir(parents=True, exist_ok=True)
        student_smoothness.to_csv(student_smoothness_filename)
                

if __name__ == '__main__':
    main()
