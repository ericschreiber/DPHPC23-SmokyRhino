#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.hpp"

int main()
{
    // matrixC
    std::vector<std::vector<float>> matrixC = {
        {1.1, 2.2, 3.3},
        {4.4, 5.5, 6.6},
        {7.7, 8.8, 9.9}};
    DenseMatrix<float> matrixC_Dense(matrixC);
    CSRMatrix<float> matrixC_HOST(matrixC_Dense);

    // matrixA
    std::vector<std::vector<float>> matrixA = {
        {2.0, 4.0, 6.0},
        {8.0, 10.0, 12.0},
        {14.0, 16.0, 18.0}};
    DenseMatrix<float> matrixA_HOST(matrixA);

    // matrixB
    std::vector<std::vector<float>> matrixB = {
        {0.5, 1.0, 1.5},
        {2.0, 2.5, 3.0},
        {3.5, 4.0, 4.5}};
    DenseMatrix<float> matrixB_HOST(matrixB);

    // matrixcalculatedSolution
    std::vector<std::vector<float>> zeroMatrix(3, std::vector<float>(3, 0.0));
    DenseMatrix<float> matrix_Zeroes(zeroMatrix);
    CSRMatrix<float> calculatedSolution_HOST(matrix_Zeroes);

    // expectedSolution
    std::vector<std::vector<float>> solution = {
        {30.25, 83.6, 158.85},
        {297.8, 506, 763.2},
        {825.45, 1280.8, 1838.95}};
    DenseMatrix<float> expectedSolution_Dense(solution);
    CSRMatrix<float> expectedSolution_HOST(expectedSolution_Dense);

    // Call multiply and pass the multiplication function from the library
    matrixC_HOST.SDDMM(
        matrixA_HOST,
        matrixB_HOST,
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