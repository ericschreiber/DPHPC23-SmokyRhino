#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "better_naive_CSR_GPU/better_naive_CSR_SDDMM_GPU.cuh"

void test_simple_near_zeros()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    const int num_iterations = 1;

    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    better_naive_CSR_SDDMM_GPU<float>* class_to_run = new better_naive_CSR_SDDMM_GPU<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &better_naive_CSR_SDDMM_GPU<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

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
        assert(false && "Test failed! Calculated solution is not equal to the expected solution.");
    }

    return;
}

void test_small()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 0}, {2, 0}, {3, 0}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}}));
    const int num_iterations = 1;

    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    better_naive_CSR_SDDMM_GPU<float>* class_to_run = new better_naive_CSR_SDDMM_GPU<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &better_naive_CSR_SDDMM_GPU<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

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
        assert(false && "Test failed! Calculated solution is not equal to the expected solution.");
    }

    return;
}

void test_small_complex()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 4, 6}, {4, 8, 12}, {6, 12, 18}}));
    const int num_iterations = 1;

    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    better_naive_CSR_SDDMM_GPU<float>* class_to_run = new better_naive_CSR_SDDMM_GPU<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &better_naive_CSR_SDDMM_GPU<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

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
        assert(false && "Test failed! Calculated solution is not equal to the expected solution.");
    }

    return;
}

void test_complex_copy_from_coo()
{
    COOMatrix<float> sample_Matrix_coo(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 0}}));

    CSRMatrix<float> sample_Matrix(sample_Matrix_coo);
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 4, 6}, {4, 8, 12}, {6, 12, 0}}));
    const int num_iterations = 1;

    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    better_naive_CSR_SDDMM_GPU<float>* class_to_run = new better_naive_CSR_SDDMM_GPU<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &better_naive_CSR_SDDMM_GPU<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

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
        std::cout << "Expected solution is: " << std::endl;
        auto expectedValues = expectedSolution.getValues();
        for (int i = 0; i < expectedValues.size(); ++i)
        {
            std::cout << expectedValues.at(i) << " ";
        }
        std::cout << std::endl;
        assert(false && "Test failed! Calculated solution is not equal to the expected solution.");
    }

    return;
}

int main()
{
    test_simple_near_zeros();
    test_small();
    test_small_complex();
    test_complex_copy_from_coo();
    return 0;
}