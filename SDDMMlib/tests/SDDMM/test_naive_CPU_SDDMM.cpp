#include <iostream>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_CPU_SDDMM.hpp"

void test_1()
{
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{10, 9}, {8, 7}, {6, 5}, {4, 3}, {2, 1}});
    CSRMatrix<float> matrixS(std::vector<std::vector<float>>{{1, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}});
    CSRMatrix<float> calculatedSolution(std::vector<std::vector<float>>{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}});
    CSRMatrix<float> expectedSolution(std::vector<std::vector<float>>{{28, 22, 0, 0, 0}, {0, 52, 0, 0, 0}, {0, 0, 60, 0, 0}, {0, 0, 0, 52, 0}, {0, 0, 0, 0, 28}});
    // This testes the Sparsematrix.SDDMM() (for CSR matrix)

    const int num_iterations = 1;

    matrixB.transpose();

    ExecutionTimer timer = ExecutionTimer();
    naive_CPU_SDDMM<float>* class_to_run = new naive_CPU_SDDMM<float>(&timer);

    matrixS.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &naive_CPU_SDDMM<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

    delete class_to_run;
    class_to_run = nullptr;

    assert(calculatedSolution == expectedSolution && "Error: The calculated solution does not match the expected in test_naive_CPU_SDDMM");
}

void test_2()
{
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}});
    CSRMatrix<float> matrixS(std::vector<std::vector<float>>{{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}});
    CSRMatrix<float> calculatedSolution(std::vector<std::vector<float>>{{100, 100, 100, 100, 100}, {100, 100, 100, 100, 100}, {100, 100, 100, 100, 100}, {100, 100, 100, 100, 100}, {100, 100, 100, 100, 100}});
    CSRMatrix<float> expectedSolution(std::vector<std::vector<float>>{{1, 2, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}});
    // This testes the Sparsematrix.SDDMM() (for CSR matrix)

    const int num_iterations = 1;

    ExecutionTimer timer = ExecutionTimer();
    naive_CPU_SDDMM<float>* class_to_run = new naive_CPU_SDDMM<float>(&timer);

    matrixS.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &naive_CPU_SDDMM<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

    delete class_to_run;
    class_to_run = nullptr;

    assert(calculatedSolution == expectedSolution && "Error: The calculated solution does not match the expected in test_naive_CPU_SDDMM");
}

int main()
{
    test_1();
    test_2();
    std::cout << "Test CPU has run sucessfully" << std::endl;

    return 0;
}