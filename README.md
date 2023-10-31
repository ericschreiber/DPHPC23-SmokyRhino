# Smokey Rhino

## SDDMMlib
This library contains implementations for matrices and the sddmm algorithms. To build it go to /SDDMMlib/build and run (with true or false depending on if you want to use cuda)
```
cmake cmake -DUSE_CUDA=<True/False> ..
cmake --build .
```
This will create the necessary libraries and test cases. You can find the files to run in the folder.
IMPORTANT NOTE: Whenever you add or change something in here WRITE A TEST CASE and something into the Readme.
To see how to use the library read the readme in the SDDMMlib folder.

## Benchmark Pipeline
Here we will implement a pipeline to benchmark the different algorithms. We will use the SDDMMlib to do so.

## third_party
If you need third party libraries put them here. If you need to build them put the build files in the third_party/build folder.
Please import them if possible as git submodules.