#!/bin/bash
#SBATCH --job-name=profiling
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:10:00

###
#   run this script with (replacing the node{ault09/ault10} and the (test)executable with your own): 
#                       sbatch --nodelist=ault09 profiling_launch.sh \
#                       ./build/SDDMMlib/tests/SDDMM/naive_dense_dense_gpu/test_compile 
###

command="$1"
module load cuda/12.1.1 cmake/3.21.3 gcc/10.2.0
nvprof --analysis-metrics \
        --cpu-thread-tracing on \
        --print-api-trace \
        --metrics all \
        --profile-child-processes \
        --system-profiling on \
        --trace api,gpu \
        --track-memory-allocations on \
        --events all \
        --csv \
        --export-profile output%p.nvprof \
        $command > profiling.out 2>&1