# Benchmark Pipeline

## Requirements
Have your function included in `/include/implemented_classes.hpp`! Otherwise it cannot be run in the benchmark pipeline.

## How to run
0. Create matrices
1. Build the project with cmake into a build folder which is top level to the project.
2. Run the bash file `run_benchmark.sh` with the following arguments:
    - `./run_benchmark.sh <path/to/config/file> <path/to/out/dir>
3. Get the results from the out directory. The file is named `results<timestamp>.csv` 

## Config file
The config file consists of lists of tests to be executed. They are written in the format:
```
function_name, path/to/first/dense/matrix, path/to/second/dense/matrix, path/to/first/sparse/matrix
function_name, path/to/first/dense/matrix, path/to/second/dense/matrix, path/to/first/sparse/matrix
```
Comments can be added by starting a line with `#`. A sample config file is provided in `/tests`.

## Generating matrices
The script `generateMatrix.sh` can be used to generate matrices. It takes the following arguments:
```
./generateMatrix.sh <path/to/output/dir> <matrix_type> <matrix_shape> (optional: <matrix_density>)
```
```
Please put all the matrices into /data/datasets/ and create a dataset folder for each kind of matrix (dense, sparse, ultra sparse, etc.).

The script `generateMatrices.sh` will generate a number of matrices with values in a given range.

Execute generateMatrices.sh to generate test matrices. Might need to execute "chmod +x generateMatrices.sh" first. 

The parameters that will be iterated over when generating the different matrices can be modified in generateMatrices.sh.

Each matrix is being saved to it's own file since I think we will probably generate very large matrices and then I think we do not want to have multiple of those in the same file (but maybe that is not a good idea and needs to be changed). 

Resulting matrices will be saved intp generated_matrices. If this folder already contains files these files will not be deleted.


## Adding new tests
New functions must be added in the `/include/implemented_classes.hpp` file. The function must be templated and must have the following signature:
```
elif (class_name == "your_SDDMM")
{
    return new your_SDDMM<T>();
}
````
Don't forget to include your header file!