#!/bin/bash
#SBATCH --job-name=profiling
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:10:00
#SBATCH --account=g34

###
#   run this script with (replacing the node{ault09/ault10} and using the inputs for the benchmark_run.sh script
#                                                                           and writing everything on one line):
#
#               sbatch --nodelist=ault09 profiling_pipeline_scratch.sh
#               /scratch/eschreib/personal_spaces/niklas/DPHPC23-SmokyRhino/benchmark_pipeline/Pipeline/benchmark_run.sh 
#               /scratch/eschreib/personal_spaces/niklas/DPHPC23-SmokyRhino/benchmark_pipeline/tests/config.example.txt 
#               /scratch/eschreib/personal_spaces/niklas/DPHPC23-SmokyRhino/profiling/results
#
#   this script runs for max 10 minutes, so if your executable takes longer, you need to adjust the time
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
        --export-profile /scratch/eschreib/personal_spaces/niklas/DPHPC23-SmokyRhino/profiling/results/output%p.nvprof \
        $script $config $output_path > /scratch/eschreib/personal_spaces/niklas/DPHPC23-SmokyRhino/profiling/results/profiling.out 2>&1