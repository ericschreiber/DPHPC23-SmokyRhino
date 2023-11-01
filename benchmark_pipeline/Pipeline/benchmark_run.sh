#!/bin/bash

# Inputs: CONFIG_FILE_PATH, OUTPUT_FOLDER

# Example Usage:
# 1. Build everything and check that the build folder is top level
# 2. benmark_run.sh /path/to/config/file /path/to/output/folder


if [ $# -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: benchmark_run.sh CONFIG_FILE_PATH OUTPUT_FOLDER"
    exit 1
fi

CONFIG_FILE_PATH=$1
OUTPUT_FOLDER=$2
EXECUTABLE_REL_PATH="../../build/benchmark_pipeline/src/benchmark_pipeline"
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
EXECUTABLE_ABS_PATH="$SCRIPTPATH/$EXECUTABLE_REL_PATH"

# Check that the config file exists
if [ ! -f "$CONFIG_FILE_PATH" ]; then
    echo "Config file does not exist"
    exit 1
fi

# Check that the output folder exists 
if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "Output folder does not exist"
    exit 1
fi

# Execute the benchmark
$EXECUTABLE_ABS_PATH $CONFIG_FILE_PATH $OUTPUT_FOLDER