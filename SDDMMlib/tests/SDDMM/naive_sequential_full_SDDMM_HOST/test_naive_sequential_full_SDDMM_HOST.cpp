#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.hpp"

int main()
{
    std::cout << "entered the main of the test" << std::endl;

    // DenseMatrix<double> matrixA(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    // DenseMatrix<double> matrixB(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    // CSRMatrix<double> matrixC(std::vector<std::vector<double>>{{1, 2, 3}, {0, 1, 2}, {0, 1, 2, 3}});
    // CSRMatrix<double> calculatedSolution(std::vector<std::vector<double>>{{0}, {0}, {0}});
    // CSRMatrix<double> expectedSolution(std::vector<std::vector<double>>{{30, 162, 450}, {0, 1, 2}, {0, 1, 2, 3}});

    std::vector<std::vector<float>> matrixC = {
        {1, 2, 3},
        {0, 1, 2},
        {0, 1, 2, 3}
    };
    DenseMatrix<float> matrixC_Dense(matrixC);
    CSRMatrix<float> matrixC_HOST(matrixC_Dense);

    std::vector<std::vector<float>> matrixA ={
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
    };
    DenseMatrix<float> matrixA_Dense(matrixA);

    std::vector<std::vector<float>> matrixB = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}
    };
    DenseMatrix<float> matrixB_Dense(matrixB);

    std::vector<std::vector<float>> zeroMatrix(3, std::vector<float>(3, 0.0));
    DenseMatrix<float> matrix_Zeros(zeroMatrix);
    CSRMatrix<float> calculatedSolution_Host(matrix_Zeros);

    std::vector<std::vector<float>> expectedSolution = {
        {30, 0 ,0},
        {0, 162, 0},
        {0, 0, 450}
    };
    DenseMatrix<float> expectedSolution_Dense(expectedSolution);
    CSRMatrix<float> expectedSolution_Host(expectedSolution_Dense);

    std::cout << "created Matrices" << std::endl;

    

    // Call multiply and pass the multiplication function from the library
    matrixC_HOST.SDDMM(
        matrixA_Dense,
        matrixB_Dense,
        calculatedSolution_Host,
        std::bind(
            &naive_sequential_full_SDDMM_HOST<float>::SDDMM,
            naive_sequential_full_SDDMM_HOST<float>(),
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4));

    // Check if the calculated solution is equal to the expected solution
    if (calculatedSolution_Host == expectedSolution_Host)
    {
        std::cout << "Test passed!" << std::endl;
    }
    else
    {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution_Host.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
    }

    return 0;
}