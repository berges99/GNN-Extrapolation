#!/bin/bash

# Variable for storing various additional arguments
ADDITIONAL=()
# Parse script arguments
while [[ "$#" -gt 0 ]]; do
	case "$1" in
		# Specific dataset
		--dataset) DATASET="$2"; shift 2;;
		--dataset=*) DATASET="${1#*=}"; shift 1;;
		# Synthetic data
		--synthetic) SYNTHETIC="$2"; shift 2;;
		--synthetic=*) SYNTHETIC="${1#*=}"; shift 1;;
		# Directly input the full (relative) path to the data
		--filepath) FILEPATH="$2"; shift 2;;
		--filepath=*) FILEPATH="${1#*=}"; shift 1;;

		# Model to be used for the teacher/student setting
		-m|--model) MODEL="$2"; shift 2;;
		-m|--model=*) MODEL="${1#*=}"; shift 1;;

		# Handle unknown parameters
		-*|--*|*) ADDITIONAL+=("$1"); shift 1;;
	esac
done

# 

