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
    CSRMatrix<double> expectedSolution(std::vector<std::vector<double>>{{16, 20, 12}, {152, 120, 60}, {420, 304, 144}});
    //This both testes the  DenseMatrix.transpose() printing out the matrices and the the Sparsematrix.SDDMM() (for CSR matrix)

    //Matrix transpose test
    matrixA.transpose();

    std::cout << "Matrix transpose test" << std::endl;
    for (int i = 0; i < matrixA.getNumRows(); i++)
    {
        for (int j = 0; j < matrixA.getNumCols(); j++)
        {
            std::cout << matrixA.at(i, j) << " ";
        }
        std::cout << std::endl;
    }

    matrixA.transpose();

    std::cout << "Matrix transposed back" << std::endl;

    for (int i = 0; i < matrixA.getNumRows(); i++)
    {
        for (int j = 0; j < matrixA.getNumCols(); j++)
        {
            std::cout << matrixA.at(i, j) << " ";
        }
        std::cout << std::endl;
    }

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
        std::cout << "CPU SDDMM test passed" << std::endl;
    }
    else
    {
        std::cout << "Test did not pass" << std::endl;
    }
    std::cout << "Test has run" << std::endl;

    return 0;
}