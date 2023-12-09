#!/bin/bash

# Path to the executable
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXECUTABLE=$SCRIPT_DIR"/../../build/benchmark_pipeline/src/generateMatrices"
DATASET_DIR=$SCRIPT_DIR"/../data/matrices"

# Yell at user if they don't provide the correct number of arguments
if [ "$#" -lt 6 ];
then
    echo "Not enough arguments. Usage: 
    ./generateMatrices.sh <MIN_N> <MAX_N> <MIN_M> <MAX_M> <min_sparsity> <max_sparsity> <stride>
    - both sparsities integers that represent percentages
    - sparsity = 100 means empty matrix, sparsity = 0 means dense matrix
    - if <stride> is not provided, it defaults to 1"
    exit 1
fi

# SET UP VARIABLES
MIN_N=$1
MAX_N=$2
MIN_M=$3
MAX_M=$4
MIN_sparsity=$5
MAX_sparsity=$6  
# if stride is not provided, it defaults to 1
if [ -z "$7" ]
then
    STRIDE=1
else 
    STRIDE=$7
fi

# Check that stride divides (max_sparsity - min_sparsity)
DIFF=$(($MAX_sparsity - $MIN_sparsity))
REST=$(($DIFF % $STRIDE))
if [ $REST -ne 0 ]
then
    echo "<stride> has to divide (<max_sparsity> - <min_sparsity>)"
    exit 1
fi

# Loop through different values of n and m
for ((n = MIN_N; n <= MAX_N; n++)); do
    for ((m = MIN_M; m <= MAX_M; m++)); do
        for ((s = MIN_sparsity; s <= MAX_sparsity; s+=$STRIDE)); do
            # if n or m is 0, skip
            if [ $n -eq 0 ] || [ $m -eq 0 ]
            then
                continue
            fi
            sparsity=$(awk -v s="$s" 'BEGIN {printf "%.2f", s / 100}')
            NAME="n_"$n"_m_"$m"_sparsity_"$sparsity
            echo "n = "$n", m = "$m", sparsity = "$sparsity
            FILE=$DATASET_DIR"/"$NAME".txt"
            touch $FILE
            $EXECUTABLE $n $m $sparsity $FILE
        done
    done
done