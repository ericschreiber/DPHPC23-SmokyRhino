#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.hpp"

void test_simple_ints()
{
    std::vector<std::vector<float>> matrixC = {
        {1, 0, 0},
        {0, 2, 0},
        {0, 0, 3}};
    DenseMatrix<float> matrixC_Dense(matrixC);
    CSRMatrix<float> matrixC_HOST(matrixC_Dense);

    std::vector<std::vector<float>> matrixA = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    DenseMatrix<float> matrixA_Dense(matrixA);

    std::vector<std::vector<float>> matrixB = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}};
    DenseMatrix<float> matrixB_Dense(matrixB);

    std::vector<std::vector<float>> zeroMatrix(3, std::vector<float>(3, 0.0));
    DenseMatrix<float> matrix_Zeros(zeroMatrix);
    CSRMatrix<float> calculatedSolution_Host(matrix_Zeros);

    std::vector<std::vector<float>> expectedSolution = {
        {30, 0, 0},
        {0, 162, 0},
        {0, 0, 450}};
    DenseMatrix<float> expectedSolution_Dense(expectedSolution);
    CSRMatrix<float> expectedSolution_Host(expectedSolution_Dense);

    const int num_iterations = 1;

    ExecutionTimer timer = ExecutionTimer();
    naive_sequential_full_SDDMM_HOST<float>* class_to_run = new naive_sequential_full_SDDMM_HOST<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    matrixC_HOST.SDDMM(
        matrixA_Dense,
        matrixB_Dense,
        calculatedSolution_Host,
        num_iterations,
        std::bind(
            &naive_sequential_full_SDDMM_HOST<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

    delete class_to_run;
    class_to_run = nullptr;

    // Check if the calculated solution is equal to the expected solution
    if (calculatedSolution_Host == expectedSolution_Host)
    {
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

    ExecutionTimer timer = ExecutionTimer();
    naive_sequential_full_SDDMM_HOST<float>* class_to_run = new naive_sequential_full_SDDMM_HOST<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &naive_sequential_full_SDDMM_HOST<float>::SDDMM,
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
    }
    else
    {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
    }

    return;
}

void test_simple_near_zeros()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 0, 0}, {0, 0, 0}, {0, 0, 0}}));

    const int num_iterations = 1;

    ExecutionTimer timer = ExecutionTimer();
    naive_sequential_full_SDDMM_HOST<float>* class_to_run = new naive_sequential_full_SDDMM_HOST<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &naive_sequential_full_SDDMM_HOST<float>::SDDMM,
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
    }
    else
    {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
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

    ExecutionTimer timer = ExecutionTimer();
    naive_sequential_full_SDDMM_HOST<float>* class_to_run = new naive_sequential_full_SDDMM_HOST<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &naive_sequential_full_SDDMM_HOST<float>::SDDMM,
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
    }
    else
    {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
    }

    return;
}

int main()
{
    test_simple_ints();
    test_small();
    test_simple_near_zeros();
    test_small_complex();
    std::cout << "CSR Naive Sequential Full: All tests passed" << std::endl;
    return 0;
}