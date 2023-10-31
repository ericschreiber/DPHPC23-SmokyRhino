// Switching between the different classes that implement the interface SDDMM
#include "SDDMMlib.hpp"
#include "naive_SDDMM.hpp"

// Put your cuda classes here. They wont be compiled if the flag USE_CUDA is not set to 1
#ifdef USE_CUDA
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.hpp"
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
#ifdef USE_CUDA
    else if (class_name == "cuda_SDDMM")
    {
        return new naive_SDDMM_GPU<T>();
    }
#endif

    else
    {
        std::cout << "Class " << class_name << " not found." << std::endl;
        return nullptr;
    }
}
