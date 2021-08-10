#!/bin/bash

# Change working directory
cd src/

# Enable multiple dataset simulations
DATASET_PATHS=(
	"../data/synthetic/erdos_renyi/N100_n100_p0.1_1626568198"
)
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
	DATASET_FILENAME="${DATASET_PATH}/raw_networkx.pkl"
	# Fetch all the different teacher outputs for the specified dataset
	TEACHER_OUTPUTS_FILENAMES=($(ls "${DATASET_PATH}/teacher_outputs/regression/GIN/"))
	DIST_MATRIX_FILENAMES=(
		"${DATASET_PATH}/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sMaxdegree/train/full64.npz"
	)
	#	"../data/synthetic/erdos_renyi/N100_n100_p0.1_1625733973/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sAvgdegree/train/full64.npz"
	#	"../data/synthetic/erdos_renyi/N100_n100_p0.1_1625733973/node_representations/WL/hashing/d3_iOnes/dist_matrices/hamming/sConstant/train/full64.npz"
	# )
	# For each of the teacher outputs
	# for TEACHER_OUTPUTS_FILENAME in "${TEACHER_OUTPUTS_FILENAMES[@]}"; do
	TEACHER_OUTPUTS_FILENAME="hidden32_blocks3_residual0_jkFalse_preFalse__initUniform_bias0.0_lower-0.1_upper0.1"
	TEACHER_OUTPUTS_FILENAME="${DATASET_PATH}/teacher_outputs/regression/GIN/${TEACHER_OUTPUTS_FILENAME}"
	# Determine initialization
	if [[ "${TEACHER_OUTPUTS_FILENAME}" == *"Uniform"* ]]; then
		initialization1="--init uniform --lower_bound -${TEACHER_OUTPUTS_FILENAME: -3} --upper_bound ${TEACHER_OUTPUTS_FILENAME: -3}"
	else
		initialization1="--init xavier --gain ${TEACHER_OUTPUTS_FILENAME: -3}"
	fi
	initialization2="--init default"
	for initialization in "${initialization1}" "${initialization2}"; do
		# Run the GIN networks
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "GIN" --hidden_dim "64" --residual "0" --jk "False" --pre_linear "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "GIN" --hidden_dim "64" --residual "1" --jk "False" --pre_linear "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "GIN" --hidden_dim "64" --residual "0" --jk "True" --pre_linear "False"
		# Run the GCN networks
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "GCN" --hidden_dim "64" --residual "0" --jk "False" --pre_linear "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "GCN" --hidden_dim "64" --residual "1" --jk "False" --pre_linear "False"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "GCN" --hidden_dim "64" --residual "0" --jk "True" --pre_linear "False"
		# Run the SGC
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "SGC" --K "3"
		# Run the SIGN
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "SIGN" --K "3" --hidden_dim "64"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "SIGN" --K "3" --hidden_dim "64" --pre_linear "True"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "SIGN" --K "2" --hidden_dim "64"
		# Run the ChebNet
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "ChebNet" --hidden_dim "64"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "ChebNet" --normalization "sym" --hidden_dim "64"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "ChebNet" --normalization "rw" --K "3" --hidden_dim "64"
		python run_model.py --dataset_filename "${DATASET_FILENAME}" --initial_relabeling "ones" \
			                --teacher_outputs_filename "${TEACHER_OUTPUTS_FILENAME}" \
			                --num_iterations "5" --epochs "200" \
			                ${initialization} \
			                "ChebNet" --normalization "rw" --hidden_dim "64"
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
		# done
	done
done

# Change working directory back
cd ../
