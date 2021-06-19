#!/bin/bash

DATASET_FILENAME="../data/synthetic/preferential_attachment/N10_n30_m1_1624020067/raw_networkx.pkl"

# Parse script arguments
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
	case "$1" in
		# Read the pair to be used
		-s|--symbol) SYMBOL="$2"; shift 2;;
		-s=*|--symbol=*) SYMBOL="${1#*=}"; shift 1;;
		# Read the market to be used
		-m|--market) MARKET="$2"; shift 2;;
		-m=*|--market=*) MARKET="${1#*=}"; shift 1;;

		# Bool indicating whether we should update the available data or not
		--update) UPDATE="$2"; shift 2;;
		--update=*) UPDATE="${1#*=}"; shift 1;;

		# Bool indicating whether we should simulate with historical data
		--simulate) SIMULATE="$2"; shift 2;;
		--simulate=*) SIMULATE="${1#*=}"; shift 1;;

		# Bool indicating whether we should predict the future (30m stochastic) candles
		--predict) PREDICT="$2"; shift 2;;
		--predict=*) PREDICT="${1#*=}"; shift 1;;
		# Handle additional predictive model parameters
		-*|--*|*) POSITIONAL+=("$1"); shift 1;;
	esac
done


# Predict and evaluate the model if necessary
PREDICT="${PREDICT:-0}"
if [ "$PREDICT" -eq "1" ]; then
	# PREDICTIONS_FILENAME=$(python3 predict_future_candles.py --symbol "$SYMBOL" --market "$MARKET" --method_kwargs "${POSITIONAL[@]}" 2>&1 > /dev/null)
	python3 predict_future_candles.py --symbol "$SYMBOL" --market "$MARKET" --method_kwargs "${POSITIONAL[@]}"
	PREDICTIONS_FILENAME=$(head -n 1 "predictions_filename.txt")
	rm "predictions_filename.txt"
fi