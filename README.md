# Smokey Rhino

This is the official repository to the [paper "Fast SDDMM on GPUs"](DPHPC_Project_Report_Fast_SDDMM_on_GPUs.pdf), where we look at optimisations for the usage of the Sampled Dense-Dense Matrix Multiplication primitive on GPUs. 

## Building and running everything
To build create a `build` folder in this directory and build with
```
cmake -DUSE_CUDA=<0/1> -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```
Afterwards you can profile your application with the scripts in `profiling` or to benchmark the pipeline you can use the scripts in `pipeline_runscripts`. Please only use these scripts to run sth on the server. You can also build a pipeline with only one application to test this without profiling. Details on how to run these scripts can be found in the corresponding folders and files.


## SDDMMlib
This library contains implementations for matrices and the sddmm algorithms. The source code of the algorithms is under /SDDMMlib/src/SDDMM.
To build it go to /SDDMMlib/build and run (with true or false depending on if you want to use cuda). To run in debug mode add `-DCMAKE_BUILD_TYPE=DEBUG` to the cmake command.
```
cmake -DUSE_CUDA=<0/1> ..
cmake --build .
```
This will create the necessary libraries and test cases. You can find the files to run in the folder.
IMPORTANT NOTE: Whenever you add or change something in here WRITE A TEST CASE and something into the Readme.
To see how to use the library read the readme in the SDDMMlib folder.

## Benchmark Pipeline
Here we will implement a pipeline to benchmark the different algorithms. We will use the SDDMMlib to do so. To build the pipeline go to /build and run the same commands as above.

## third_party
If you need third party libraries put them here. If you need to build them put the build files in the third_party/build folder.
Please import them if possible as git submodules.

## run all tests
Run all tests that we have by executing "./tests.sh". At the end you get a result back if all tests passed or not. If not you can ckeck the output above. 
Should you create new tests please add them to the tests.sh script. Please add them to the correct section GPU/CPU.
