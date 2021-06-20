#!/bin/bash

DATASET_FILENAME="../data/synthetic/preferential_attachment/N10_n30_m1_1624020067/raw_networkx.pkl"

COUNT=1000

for i in $(seq $COUNT); do
	python3 run_teacher.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
	                       --verbose "1" --save_file_destination "0" \
	                       --setting "regression" \
	                       --bias "0" --lower_bound "-0.3" --upper_bound "0.3" \
	                       GIN --num_features "1" --hidden_dim "32" --residual "0" --jk "0"
done