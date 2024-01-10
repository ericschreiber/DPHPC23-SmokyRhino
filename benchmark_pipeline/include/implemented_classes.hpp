// Switching between the different classes that implement the interface SDDMM
#ifndef IMPLEMENTED_CLASSES_HPP
#define IMPLEMENTED_CLASSES_HPP

#include "COOMatrix.hpp"
#include "CSRMatrix.hpp"
#include "SDDMMlib.hpp"
#include "naive_CPU_SDDMM.hpp"
#include "naive_SDDMM.hpp"
#include "naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.hpp"
#include "naive_sequential_sampled_SDDMM_HOST/naive_sequential_sampled_SDDMM_HOST.hpp"

// Put your cuda classes here. They wont be compiled if the flag USE_CUDA is not set to 1
#if USE_CUDA
#include "SM_L2/SM_L2_GPU.hpp"
#include "better_naive_CSR_GPU/better_naive_CSR_SDDMM_GPU.cuh"
#include "cache_coo_gpu/cache_coo_SDDMM_GPU.hpp"
#include "coo_opt_loop_unrolled_gpu/coo_opt_loop_unrolled_SDDMM_GPU.hpp"
#include "coo_opt_vectorization_gpu/coo_opt_vectorization_SDDMM_GPU.hpp"
#include "coo_tiling_naive_gpu/coo_tiling_naive_gpu_SDDMM_GPU.hpp"
#include "cusparse_baseline/cusparse_baseline.hpp"
#include "merged/merged.hpp"
#include "naive_coo_gpu/naive_coo_SDDMM_GPU.hpp"
#include "naive_csr_via_coo_gpu/naive_csr_via_coo_SDDMM_GPU.hpp"
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.cuh"
#endif

// Get the class with the given name
template <typename T>
SDDMMlib<T>* get_implemented_SDDMM(std::string class_name)
{
    if (class_name == "naive_SDDMM")
    {
        return new naive_SDDMM<T>();
    }
    else if (class_name == "naive_sequential_sampled_SDDMM_HOST")
    {
        return new naive_sequential_sampled_SDDMM_HOST<T>();
    }
    else if (class_name == "naive_sequential_full_SDDMM_HOST")
    {
        return new naive_sequential_full_SDDMM_HOST<T>();
    }
    else if (class_name == "naive_CPU_SDDMM")
    {
        return new naive_CPU_SDDMM<T>();
    }

// Put your cuda classes here. They wont be compiled if the flag USE_CUDA is not set to 1
#if USE_CUDA
    else if (class_name == "naive_SDDMM_GPU")
    {
        return new naive_SDDMM_GPU<T>();
    }
    else if (class_name == "better_naive_CSR_SDDMM_GPU")
    {
        return new better_naive_CSR_SDDMM_GPU<T>();
    }
    else if (class_name == "naive_coo_SDDMM_GPU")
    {
        return new naive_coo_SDDMM_GPU<T>();
    }
    else if (class_name == "cache_coo_SDDMM_GPU")
    {
        return new cache_coo_SDDMM_GPU<T>();
    }
    else if (class_name == "naive_csr_via_coo_SDDMM_GPU")
    {
        return new naive_csr_via_coo_SDDMM_GPU<T>();
    }
    else if (class_name == "coo_opt_vectorization_SDDMM_GPU")
    {
        return new coo_opt_vectorization_SDDMM_GPU<T>();
    }
    else if (class_name == "coo_opt_loop_unrolled_SDDMM_GPU")
    {
        return new coo_opt_loop_unrolled_SDDMM_GPU<T>();
    }
    else if (class_name == "coo_tiling_naive_gpu_SDDMM_GPU")
    {
        return new coo_tiling_naive_gpu_SDDMM_GPU<T>();
    }
    else if (class_name == "merged")
    {
        return new merged<T>();
    }
    else if (class_name == "sml2_paper")
    {
        return new sm_l2_SDDMM_GPU<T>();
    }
    else if (class_name == "cusparse_baseline")
    {
        return new cusparse_baseline<T>();
    }

#endif

    else
    {
        std::cout << "Class " << class_name << " not found." << std::endl;
        assert(false && "Error: Class not found.");
        return nullptr;
    }
}

template <typename T, typename... Args>
SparseMatrix<T>* get_implemented_SparseMatrix_from_coo(std::string class_name, Args&&... constructorArgs)
{
    if (class_name == "CSRMatrix")
    {
        return new CSRMatrix<T>(std::forward<Args>(constructorArgs)...);
    }
    else if (class_name == "COOMatrix")
    {
        return new COOMatrix<T>(std::forward<Args>(constructorArgs)...);
    }
    else
    {
        std::cout << "Class " << class_name << " not found." << std::endl;
        assert(false && "Error: Class not found.");
        return nullptr;
    }
}
#endif  // IMPLEMENTED_CLASSES_HPP