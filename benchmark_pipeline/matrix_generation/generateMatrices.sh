#!/bin/bash

# Path to the executable
EXECUTABLE="./generateMatrices"

# PARAMETER RANGES
MIN_N=5
MAX_N=10
MIN_M=5
MAX_M=10
MIN_sparsity=0
MAX_sparsity=1

# Loop through different values of n and m
for ((n = MIN_N; n <= MAX_N; n++)); do
    for ((m = MIN_M; m <= MAX_M; m++)); do
        for ((s = MIN_sparsity; s <= MAX_sparsity; s++)); do
            NAME="n_"$n"_m_"$m"_sparsity_"$s
            touch "generated_matrices/"$NAME".txt"
            $EXECUTABLE $n $m $s "generated_matrices/"$NAME".txt"
        done
    done
done