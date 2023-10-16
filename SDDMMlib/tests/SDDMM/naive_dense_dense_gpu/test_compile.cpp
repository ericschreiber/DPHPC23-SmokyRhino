#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.hpp"

int main()
{
    CSRMatrix<float> matrixA_HOST(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    DenseMatrix<float> matrixB_HOST(std::vector<std::vector<float>>{{1}, {2}, {3}});
    DenseMatrix<float> matrixC_HOST(std::vector<std::vector<float>>{{1}, {2}, {3}});
    CSRMatrix<float> calculatedSolution_HOST(std::vector<std::vector<float>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    CSRMatrix<float> expectedSolution_HOST(std::vector<std::vector<float>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});

    // Call multiply and pass the multiplication function from the library
    matrixA_HOST.SDDMM(
        matrixB_HOST,
        matrixC_HOST,
        calculatedSolution_HOST,
        std::bind(
            &naive_SDDMM_GPU<float>::SDDMM,
            naive_SDDMM_GPU<float>(),
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4));

    // Check if the calculated solution is equal to the expected solution
    if (calculatedSolution_HOST == expectedSolution_HOST)
    {
        std::cout << "Test passed!" << std::endl;
    }
    else
    {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution_HOST.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
    }

    return 0;
}