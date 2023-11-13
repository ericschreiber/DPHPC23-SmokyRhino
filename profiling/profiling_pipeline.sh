#!/bin/bash
#SBATCH --job-name=profiling
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:10:00

###
#   run this script with (replacing the node{ault09/ault10} and using the inputs for the benchmark_run.sh script): 
#                       sbatch --nodelist=ault09 profiling_launch.sh \
#                       /users/eschreib/niklas/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/benchmark_run.sh 
#                       /users/eschreib/niklas/DPHPC23-SmokyRhino/benchmark_pipeline/tests/config.example.txt 
#                       /users/eschreib/niklas/DPHPC23-SmokyRhino/ 
###

script="$1"
config="$2"
output_path="$3"
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
        $script $config $output_path > profiling.out 2>&1