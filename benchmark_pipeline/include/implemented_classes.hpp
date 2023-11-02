// Switching between the different classes that implement the interface SDDMM
#include "SDDMMlib.hpp"
#include "naive_SDDMM.hpp"
#include "naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.hpp"
#include "naive_sequential_sampled_SDDMM_HOST/naive_sequential_sampled_SDDMM_HOST.hpp"

// Get the class with the given name
template <typename T>
SDDMMlib<T>* get_implemented_class(std::string class_name)
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
    else
    {
        std::cout << "Class " << class_name << " not found." << std::endl;
        return nullptr;
    }
}
