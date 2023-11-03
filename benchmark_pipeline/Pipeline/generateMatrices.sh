#!/bin/bash

# Path to the executable
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXECUTABLE=$SCRIPT_DIR"/../../build/benchmark_pipeline/src/generateMatrices"
DATASET_DIR=$SCRIPT_DIR"/../data/matrices"

# Yell at user if they don't provide the correct number of arguments
if [ "$#" -ne 6 ];
then
    echo "Usage: 
    ./generateMatrices.sh <MIN_N> <MAX_N> <MIN_M> <MAX_M> <min_sparsity> <max_sparsity>
    (both sparsities integers that represent percentages)"
    exit 1
fi

# PARAMETER RANGES
MIN_N=$1
MAX_N=$2
MIN_M=$3
MAX_M=$4
MIN_sparsity=$5
MAX_sparsity=$6  

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