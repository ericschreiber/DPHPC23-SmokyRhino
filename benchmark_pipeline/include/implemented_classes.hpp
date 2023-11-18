// Switching between the different classes that implement the interface SDDMM
#ifndef IMPLEMENTED_CLASSES_HPP
#define IMPLEMENTED_CLASSES_HPP

#include "SDDMMlib.hpp"
#include "naive_SDDMM.hpp"

// Put your cuda classes here. They wont be compiled if the flag USE_CUDA is not set to 1
#if USE_CUDA
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.cuh"
#include "semi_naive_CSR_SDDMM_GPU.cuh"
#endif

// Get the class with the given name
template <typename T>
SDDMMlib<T>* get_implemented_class(std::string class_name)
{
    if (class_name == "naive_SDDMM")
    {
        return new naive_SDDMM<T>();
    }

// Put your cuda classes here. They wont be compiled if the flag USE_CUDA is not set to 1
#if USE_CUDA
    else if (class_name == "naive_SDDMM_GPU")
    {
        return new naive_SDDMM_GPU<T>();
    }
    else if (class_name == "semi_naive_CSR_SDDMM_GPU")
    {
        return new semi_naive_CSR_SDDMM_GPU<T>();
    }
#endif

    else
    {
        std::cout << "Class " << class_name << " not found." << std::endl;
        assert(false && "Error: Class not found.");
        return nullptr;
    }
}

#endif  // IMPLEMENTED_CLASSES_HPP