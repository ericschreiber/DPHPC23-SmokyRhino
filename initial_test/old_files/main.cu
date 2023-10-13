#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

#include "utils.h"
#include "CUDATimer.cuh"

void mysampling(int, const double*, double*);
double rmse(int, const double*, const double*);

int main()
{
  int steps = 10;

  // Set-up host-side matrices, must be multiple of BLOCK_SIZE and BLOCK_ITEMS_XY
  // from my_dgemm_*.cu files
  int m = 1 << 13;
  int n = 1 << 12;
  int k = 1 << 12;

  std::mt19937 gen{42};
  std::uniform_real_distribution<double> unif(-1., 1.);
  double* h_A;
  double* h_B;
  double* h_C;
  CUDA_CHECK(cudaMallocHost(&h_A, m*k*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&h_B, k*n*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&h_C, m*n*sizeof(double)));

  for (int i(0); i < m*k; ++i)
    h_A[i] = unif(gen);
  for (int i(0); i < k*n; ++i)
    h_B[i] = unif(gen);
  for (int i(0); i < m*n; ++i)
    h_C[i] = unif(gen);

  // GEMM coefficients
  double alpha = unif(gen);
  double beta = unif(gen);

  // Set-up streams and handles
  cudaStream_t s;
  cublasHandle_t handle;
  CUDA_CHECK(cudaStreamCreate(&s));
  CUDA_CHECK(cublasCreate(&handle));

  // Allocate device memory and copy matrices from host
  double* d_A;
  double* d_B;
  double* d_C;
  double* d_D;
  double* d_Ccublas;
  CUDA_CHECK(cudaMalloc(&d_A, m*k*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_B, k*n*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_C, m*n*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_D, m*n*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Ccublas, m*n*sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, m*k*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, k*n*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, h_C, m*n*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_D, h_C, m*n*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Ccublas, h_C, m*n*sizeof(double), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaProfilerStart()); // if you want to use nvprof
  // Correctness check between cublasDgemm and myDgemm
  //CUDA_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_Ccublas, m));
  //myDgemm(m, n, k, alpha, d_A, d_B, beta, d_C);
  //CUDA_CHECK(cudaDeviceSynchronize());
  //double myRmse = rmse(m*n, d_Ccublas, d_C);
  //std::cout << "My GEMM :   RMSE(Ccublas,C) = " << myRmse << std::endl;
  //CUDA_CHECK(cudaProfilerStop()); // end of profiling region

  // Benchmark Dgemm
  EventTimer timer;

  // 1) Warmup
  CUDA_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_Ccublas, m));

  timer.start(0);
  for (int i=0; i < steps; i++)
    CUDA_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_Ccublas, m));

  timer.stop(0); // implicit synchronization within this function
  float cublasAvg = timer.elapsed() / (float)steps;
  std::cout << "cublasDgemm average runtime : " << cublasAvg << "[ms]" << std::endl;

  // 2) Warmup
  timer.start(0);
  for (int i=0; i < steps; i++)
  ////////////////////////////////////////////////////////////////////////////////////////////
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_Ccublas, m);
    mysampling(m*n, d_C, d_D);
  ////////////////////////////////////////////////////////////////////////////////////////////
  timer.stop(0); // implicit synchronization within this function

  // Launch with Dgemm with cuBLAS to benchmark against
  float myAvg = timer.elapsed() / (float)steps;
  std::cout << "myDgemm     average runtime : " << myAvg << "[ms]" << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_D));
  CUDA_CHECK(cudaFree(d_Ccublas));
  CUDA_CHECK(cublasDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(s));

  return 0;
}


double rmse(const int n, const double* const dVec1, const double* const dVec2)
{
  double* hVec1;
  double* hVec2;
  CUDA_CHECK(cudaMallocHost(&hVec1, n*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&hVec2, n*sizeof(double)));
  
  CUDA_CHECK(cudaMemcpy(hVec1, dVec1, n*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hVec2, dVec2, n*sizeof(double), cudaMemcpyDeviceToHost));

  double rmse = 0.;
  for (int i(0); i < n; ++i)
  {
    double diff = hVec1[i] - hVec2[i];
    rmse += (diff*diff);
  }

  CUDA_CHECK(cudaFreeHost(hVec1));
  CUDA_CHECK(cudaFreeHost(hVec2));

  return std::sqrt(rmse/n);
}
