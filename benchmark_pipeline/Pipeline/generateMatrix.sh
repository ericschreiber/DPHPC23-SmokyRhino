#!/bin/bash

# Inputs: OUTPUT_DIR, MATRIX_SHAPE, MATRIX_DENSITY (optional and in 0.1 or 1 notation)

# If density is 1 or not specified, the matrix will be fully dense and stored in dense format. 
# The A & B Matrices need to be in dense format for the dense matrix multiplication benchmark.

# Example Usage:
# ./generateMatrix.sh <path/to/output/dir> <matrix_shape> (optional: <matrix_density>)
# ./generateMatrix.sh ../data/matrices "5x5"


if [ $# -ne 3 ] && [ $# -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "Usage: generateMatrix.sh OUTPUT_DIR MATRIX_SHAPE (optional: MATRIX_DENSITY)"
    exit 1
fi

OUTPUT_DIR=$1
MATRIX_SHAPE=$2
if [ $# -eq 3 ]; then
    MATRIX_DENSITY=$3
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


# if the output directory does not exist, create it
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi
# remove the / at the end of the output directory if it exists
if [[ $OUTPUT_DIR == */ ]]; then
    OUTPUT_DIR=${OUTPUT_DIR::-1}
fi
FILE=$OUTPUT_DIR"/"$NAME
# if density is not 1 then the matrix is sparse and stored in mtx formats
if [ $MATRIX_DENSITY != 1 ]; then
    FILE=$FILE".mtx"
fi
touch $FILE


$EXECUTABLE "${n}" "${m}" "${MATRIX_DENSITY}" "${FILE}"
echo "Generated matrix "$FILE