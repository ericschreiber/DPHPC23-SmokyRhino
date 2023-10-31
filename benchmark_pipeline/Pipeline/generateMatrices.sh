#!/bin/bash

# Path to the executable
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXECUTABLE=$SCRIPT_DIR"/../../build/benchmark_pipeline/src/generateMatrices"
DATASET_DIR=$SCRIPT_DIR"/../data/matrices"

# PARAMETER RANGES
MIN_N=5
MAX_N=5
MIN_M=5
MAX_M=5
MIN_sparsity=0
MAX_sparsity=0  # in percent

# Loop through different values of n and m
for ((n = MIN_N; n <= MAX_N; n++)); do
    for ((m = MIN_M; m <= MAX_M; m++)); do
        for ((s = MIN_sparsity; s <= MAX_sparsity; s++)); do
            sparsity=$(echo "scale=2; $s/100" | bc)
            NAME="n_"$n"_m_"$m"_sparsity_"$sparsity
            FILE=$DATASET_DIR"/"$NAME".txt"
            touch $FILE
            $EXECUTABLE $n $m $sparsity $FILE
            echo "Generated matrix "$FILE
        done
    done
done