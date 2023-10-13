#!/bin/bash
rm -f *.o *.exe
nvcc -c main.cu -o main.o -I../../common/inc -I../include -O3 -std=c++11 \
                    --compiler-options "-Wall -Wextra" -use_fast_math -lcublas \
                    -gencode=arch=compute_86,code=sm_86 -lineinfo
nvcc -c sampling.cu -o sampling.o -I../../common/inc -I../include -O3 -std=c++11 \
                    --compiler-options "-Wall -Wextra" -use_fast_math -lcublas \
                    -gencode=arch=compute_86,code=sm_86 -lineinfo
nvcc main.o sampling.o -o execute.exe -O3 -std=c++11 \
                    --compiler-options "-Wall -Wextra" -use_fast_math -lcublas \
                    -gencode=arch=compute_86,code=sm_86 -lineinfo