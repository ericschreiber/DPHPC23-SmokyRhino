# Smokey Rhino

## SDDMMlib
This library contains implementations for matrices and the sddmm algorithms. To build it go to /SDDMMlib/build and run (with true or false depending on if you want to use cuda)
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
run all tests that we have by executing "./tests.sh". Each test executable should then print out something along the lines of "success running all tests for class X" (that's how you can check if all tests passed). 
should you create new tests that do not get run by "tests.sh" you have to add the path to the folder (make sure the only executables in this folder are tests because all excutables will be run) where the tests sit to the script. 