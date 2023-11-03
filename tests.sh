#!/bin/bash

# add more test directories here if we use more. 
# make sure that the dir only contains test executables because all executables in the dir will be run.
dirs=("./build/SDDMMlib/tests/MatrixLib")

# Loop through all directories
for dir in "${dirs[@]}"; do
  # Loop through all files in the directory
  for file in "$dir"/*; do
    # skip is file is a folder
    if [[ -d "$file" ]]; then
      continue
    fi
    # If the file is executable
    if [[ -x "$file" ]]; then
      # Run the file
      "$file"
    # echo a newline
    echo ""
    fi
  done
done