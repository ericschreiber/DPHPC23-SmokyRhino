#include <cuda_runtime.h>
#include "utils.h"

#define BLOCK_SIZE 32

__global__ void naivesampling(
    const int m,
    const int n,
    const double* const A,
    double* const C)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    C[col*m + row] = C[col*m + row] * A[col*m + row];
}

void mysampling(
    const int m,
    const int n,
    const double* const A,
    double* const C)
{
  dim3 dimGrid(n/BLOCK_SIZE, m/BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  naivesampling<<<dimGrid, dimBlock>>>(m, n, A, C);
  CUDA_CHECK(cudaGetLastError());
}