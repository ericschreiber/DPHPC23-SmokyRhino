#include <cassert>
#include <iostream>
#include <vector>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_coo_gpu/naive_coo_SDDMM_GPU.hpp"

void test_simple_zeros()
{
    COOMatrix<float> matrixA(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1}, {2}, {3}});
    DenseMatrix<float> matrixC(std::vector<std::vector<float>>{{1}, {2}, {3}});
    COOMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}));
    COOMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}));

    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    naive_coo_SDDMM_GPU<float>* class_to_run = new naive_coo_SDDMM_GPU<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    matrixA.SDDMM(
        matrixB,
        matrixC,
        calculatedSolution,
        std::bind(
            &naive_coo_SDDMM_GPU<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4));

    delete class_to_run;
    class_to_run = nullptr;

    // Check if the calculated solution is equal to the expected solution
    if (calculatedSolution == expectedSolution)
    {
        std::cout << "Test passed!" << std::endl;
    }
    else
    {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
        std::cout << std::endl;
    }

    return;
}

int main()
{
    test_simple_zeros();
    return 0;
}