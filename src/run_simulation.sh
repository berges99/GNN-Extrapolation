#!/bin/bash

DATASET_FILENAME="../data/synthetic/erdos_renyi/N300_n100_p0.1_1624629108/raw_networkx.pkl"
TEACHER_OUTPUTS_FILENAME="../data/synthetic/erdos_renyi/N300_n100_p0.1_1624629108/teacher_outputs/regression/GIN/hidden32_blocks3_residualFalse_jkTrue_mlp2__initXavier_bias0.0_gain2.0"

DIST_FULL="../data/synthetic/erdos_renyi/N300_n100_p0.1_1624629108/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sMaxdegree/full64.npz"
DIST_NYSTROM="../data/synthetic/erdos_renyi/N300_n100_p0.1_1624629108/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sMaxdegree/nystrom64_1625141663830.npz"


# Run the GIN models
python3 run_model.py --dataset_filename $DATASET_FILENAME --initial_relabeling "ones" \
                     --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                     --init "xavier" --gain "2.0" \
                     "GIN" --hidden_dim "64" --mlp "2" --jk "False" --residual "False"
python3 run_model.py --dataset_filename $DATASET_FILENAME --initial_relabeling "ones" \
                     --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                     --init "xavier" --gain "2.0" \
                     "GIN" --hidden_dim "64" --mlp "2" --jk "True" --residual "False"                    
python3 run_model.py --dataset_filename $DATASET_FILENAME --initial_relabeling "ones" \
                     --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                     --init "xavier" --gain "2.0" \
                     "GIN" --hidden_dim "64" --mlp "2" --jk "False" --residual "True"

# Run the Baseline models
python3 run_model.py --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                     "Baseline"  --dist_matrix_filename $DIST_FULL --method "knn"
python3 run_model.py --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                     "Baseline"  --dist_matrix_filename $DIST_FULL --method "baseline" 
python3 run_model.py --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                     "Baseline"  --dist_matrix_filename $DIST_NYSTROM --method "knn"                                        
python3 run_model.py --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                     "Baseline"  --dist_matrix_filename $DIST_NYSTROM --method "baseline" 

# Run smoothness
python3 run_smoothness.py --dist_matrix_filename $DIST_FULL \
                          --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                          --verbose "True"
python3 run_smoothness.py --dist_matrix_filename $DIST_NYSTROM \
                          --teacher_outputs_filename $TEACHER_OUTPUTS_FILENAME \
                          --verbose "True"                          