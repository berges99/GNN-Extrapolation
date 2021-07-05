#!/bin/bash

# Change working directory
cd src/

DATASET_FILENAME="../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/raw_networkx.pkl"
TEACHER_OUTPUTS_FILENAMES=($(ls "../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/teacher_outputs/regression/GIN/"))
DIST_MATRIX_FILENAMES=(
	"../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sMaxdegree/train/full64.npz"
	"../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sAvgdegree/train/full64.npz"
	"../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sConstant/train/full64.npz"
	"../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/node_representations/WL/continuous/d3_iOnes_nWasserstein/dist_matrices/l2/train/full64.npz"
)

# For each of the teacher outputs
for TEACHER_OUTPUTS_FILENAME in "${TEACHER_OUTPUTS_FILENAMES[@]}"; do
	TEACHER_OUTPUTS_FILENAME="../data/synthetic/erdos_renyi/N100_n100_p0.1_1625478135/teacher_outputs/regression/GIN/${TEACHER_OUTPUTS_FILENAMES}"
	echo $TEACHER_OUTPUTS_FILENAME
	# Run the Networks with different uniform initializations
	for limit in "0.1" "0.2" "0.3" "0.5" "1.0"; do
		# Run the GIN networks
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "GIN" --hidden_dim "64" --residual "False" --jk "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "GIN" --hidden_dim "64" --residual "False" --jk "True"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "GIN" --hidden_dim "64" --residual "True" --jk "False"
		# Run the GCN networks
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "GCN" --hidden_dim "64" --residual "False" --jk "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "GCN" --hidden_dim "64" --residual "False" --jk "True"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "GCN" --hidden_dim "64" --residual "True" --jk "False"
		# Run the SGC
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "SGC" --K "3"
		# Run the SIGN
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "SIGN" --K "3"
		# Run the ChebNet
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "ChebNet"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "ChebNet" --normalization "sym"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "uniform" --lower_bound "-${limit}" --upper_bound "${limit}" \
			                "ChebNet" --normalization "rw"
	done
	# Run the Networks with different Xavier uniform initializations
	for gain in "1.0" "2.0"; do
		# Run the GIN networks
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "GIN" --hidden_dim "64" --residual "False" --jk "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "GIN" --hidden_dim "64" --residual "False" --jk "True"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "GIN" --hidden_dim "64" --residual "True" --jk "False"
		# Run the GCN networks
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "GCN" --hidden_dim "64" --residual "False" --jk "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "GCN" --hidden_dim "64" --residual "False" --jk "True"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "GCN" --hidden_dim "64" --residual "True" --jk "False"
		# Run the SGC
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "SGC" --K "3"
		# Run the SIGN
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "SIGN" --K "3"
		# Run the ChebNet
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "ChebNet"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "ChebNet" --normalization "sym"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --init "xavier" --gain "${gain}" \
			                "ChebNet" --normalization "rw"
	done
	##########
	# For the different distance matrices
	for dist_matrix_filename in "${DIST_MATRIX_FILENAMES[@]}"; do
		# Run the baselines (# TBD add smoothed versions)
		for method in "baseline" "knn"; do
			python run_model.py --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                    "Baseline" --dist_matrix_filename "${dist_matrix_filename}" \
			                    --method "${method}"
		done
		# Compute the smoothness
		python run_smoothness.py --dist_matrix_filename "${dist_matrix_filename}" \
		                         --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
		                         --verbose "True"
	done
done

# Change working directory back
cd ../
