#!/bin/bash

# Inputs: OUTPUT_DIR, MATRIX_TYPE, MATRIX_SHAPE, MATRIX_DENSITY (optional and in 0.1 or 1 notation)

# Example Usage:
# ./generateMatrix.sh <path/to/output/dir> <matrix_type> <matrix_shape> (optional: <matrix_density>)
# ./generateMatrix.sh ../data/matrices dense "5x5"


if [ $# -ne 3 ] && [ $# -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage: generateMatrix.sh OUTPUT_DIR MATRIX_TYPE MATRIX_SHAPE (optional: MATRIX_DENSITY)"
    exit 1
fi

OUTPUT_DIR=$1
MATRIX_TYPE=$2
MATRIX_SHAPE=$3
if [ $# -eq 4 ]; then
    MATRIX_DENSITY=$4
else 
    MATRIX_DENSITY=1
fi

# Path to the executable
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXECUTABLE=$SCRIPT_DIR"/../../build/benchmark_pipeline/src/generateMatrices"
DATASET_DIR=$SCRIPT_DIR"/../data/matrices"

# Get the dimensions of the matrix
n=$(echo $MATRIX_SHAPE | cut -d'x' -f1)
m=$(echo $MATRIX_SHAPE | cut -d'x' -f2)
# check that there are no non-numeric characters in the dimensions
if ! [[ $n =~ ^[0-9]+$ ]] || ! [[ $m =~ ^[0-9]+$ ]]; then
    echo "Matrix shape must be of the form <int>x<int>"
    exit 1
fi

# Remove the point in the density if it exists
MATRIX_DENSITY_file_name=$(echo $MATRIX_DENSITY | sed 's/\.//g')
NAME="n_"$n"_m_"$m"_sparsity_"$MATRIX_DENSITY_file_name
FILE=$OUTPUT_DIR"/"$NAME
touch $FILE
$EXECUTABLE "${n}" "${m}" "${MATRIX_DENSITY}" "${FILE}"
echo "Generated matrix "$FILE