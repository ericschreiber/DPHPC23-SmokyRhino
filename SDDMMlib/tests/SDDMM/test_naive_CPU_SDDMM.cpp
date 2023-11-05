#include <iostream>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_CPU_SDDMM.hpp"

int main()
{
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{10, 9}, {8, 7}, {6, 5}, {4, 3}, {2, 1}});
    CSRMatrix<float> matrixS(std::vector<std::vector<float>>{{1, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}});
    CSRMatrix<float> calculatedSolution(std::vector<std::vector<float>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}});
    CSRMatrix<float> expectedSolution(std::vector<std::vector<float>>{{28, 22, 0, 0, 0}, {0, 52, 0, 0, 0}, {0, 0, 60, 0, 0}, {0, 0, 0, 52, 0}, {0, 0, 0, 0, 28}});
    // This testes the Sparsematrix.SDDMM() (for CSR matrix)

    ExecutionTimer timer = ExecutionTimer();
    naive_CPU_SDDMM<float>* class_to_run = new naive_CPU_SDDMM<float>(&timer);

    matrixS.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        std::bind(
            &naive_CPU_SDDMM<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4));

    delete class_to_run;
    class_to_run = nullptr;

    assert(calculatedSolution == expectedSolution && "Error: The calculated solution does not match the expected in test_naive_CPU_SDDMM");

    if (calculatedSolution == expectedSolution)
    {
        std::cout << "naive CPU SDDMM test passed" << std::endl;
    }
    else
    {
        std::cout << "Test did not pass" << std::endl;
    }
    std::cout << "Test has run" << std::endl;

    return 0;
}