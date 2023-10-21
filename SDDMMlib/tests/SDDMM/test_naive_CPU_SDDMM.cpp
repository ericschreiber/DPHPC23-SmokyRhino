#include <iostream>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_CPU_SDDMM.hpp"

int main()
{
    DenseMatrix<double> matrixA(std::vector<std::vector<double>>{{1, 2}, {3, 4}, {5, 6}});
    DenseMatrix<double> matrixB(std::vector<std::vector<double>>{{6, 5}, {4, 3}, {2, 1}});
    CSRMatrix<double> matrixS(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    CSRMatrix<double> calculatedSolution(std::vector<std::vector<double>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    CSRMatrix<double> expectedSolution(std::vector<std::vector<double>>{{84, 114, 144}, {204, 276, 348}, {324, 438, 552}});

    matrixS.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        std::bind(
            &naive_CPU_SDDMM<double>::SDDMM,
            naive_CPU_SDDMM<double>(),
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4));

        if (calculatedSolution == expectedSolution)
    {
        std::cout << "Test passed" << std::endl;
    }
    else
    {
        std::cout << "Test did not pass" << std::endl;
    }
    std::cout << "Test has run" << std::endl;
    return 0;
}