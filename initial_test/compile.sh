#!/bin/bash
rm -f *.o *.exe
nvcc -c naive_dense_dense.cu -o naive_dense_dense.o -I../../common/inc -I../include -O3 -std=c++11 \
                    --compiler-options "-Wall -Wextra" -use_fast_math -lcublas \
                    -gencode=arch=compute_86,code=sm_86 -lineinfo
nvcc -c my_naive_sampling.cu -o my_naive_sampling.o -I../../common/inc -I../include -O3 -std=c++11 \
                    --compiler-options "-Wall -Wextra" -use_fast_math -lcublas \
                    -gencode=arch=compute_86,code=sm_86 -lineinfo
nvcc naive_dense_dense.o my_naive_sampling.o -o execute.exe -O3 -std=c++11 \
                    --compiler-options "-Wall -Wextra" -use_fast_math -lcublas \
                    -gencode=arch=compute_86,code=sm_86 -lineinfo