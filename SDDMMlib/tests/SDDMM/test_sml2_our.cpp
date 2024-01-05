#include <cassert>
#include <iostream>
#include <vector>

#include "COOMatrix.hpp"
#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "sml2_our/sml2.cuh"

void run_testcase(CSRMatrix<float> sample_Matrix, DenseMatrix<float> matrixA, DenseMatrix<float> matrixB, CSRMatrix<float> calculatedSolution, CSRMatrix<float> expectedSolution)
{
    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    sml2_our<float>* class_to_run = new sml2_our<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        1,
        std::bind(
            &sml2_our<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

    delete class_to_run;
    class_to_run = nullptr;

    // print calculated solution
    std::cout << "Calculated solution is: " << std::endl;
    auto calculatedValues = calculatedSolution.getValues();
    for (int i = 0; i < calculatedValues.size(); ++i)
    {
        std::cout << calculatedValues.at(i) << " ";
    }
    std::cout << std::endl;

    // // Check if the calculated solution is equal to the expected solution
    // if (calculatedSolution == expectedSolution)
    // {
    //     std::cout << "Test passed!" << std::endl;
    //     std::cout << std::endl;
    // }
    // else
    // {
    //     std::cout << "Test failed! Calculated solution is: " << std::endl;
    //     auto calculatedValues = calculatedSolution.getValues();
    //     for (int i = 0; i < calculatedValues.size(); ++i)
    //     {
    //         std::cout << calculatedValues.at(i) << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << "Expected solution is: " << std::endl;
    //     auto expectedValues = expectedSolution.getValues();
    //     for (int i = 0; i < expectedValues.size(); ++i)
    //     {
    //         std::cout << expectedValues.at(i) << " ";
    //     }
    //     std::cout << std::endl;
    //     std::cout << std::endl;
    // }

    return;
}

void t1()
{
    CSRMatrix<float> sample_Matrix(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}}));
    DenseMatrix<float> matrixA(
        std::vector<std::vector<float>>{
            {1, 2, 3, 4, 5, 6, 7, 8},
            {9, 10, 11, 12, 13, 14, 15, 16},
            {17, 18, 19, 20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29, 30, 31, 32}});
    DenseMatrix<float> matrixB(
        std::vector<std::vector<float>>{
            {1, 9, 17, 25},
            {2, 10, 18, 26},
            {3, 11, 19, 27},
            {4, 12, 20, 28},
            {5, 13, 21, 29},
            {6, 14, 22, 30},
            {7, 15, 23, 31},
            {8, 16, 24, 32}});
    CSRMatrix<float> calculatedSolution(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}}));
    CSRMatrix<float> expectedSolution(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

void t2()
{
    CSRMatrix<float> sample_Matrix(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {1, 0, 0, 4},
                {5, 6, 0, 0},
                {0, 0, 0, 0},
                {13, 14, 15, 16}}));
    DenseMatrix<float> matrixA(
        std::vector<std::vector<float>>{
            {1, 2, 3, 4, 5, 6, 7, 8},
            {9, 10, 11, 12, 13, 14, 15, 16},
            {17, 18, 19, 20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29, 30, 31, 32}});
    DenseMatrix<float> matrixB(
        std::vector<std::vector<float>>{
            {1, 9, 17, 25},
            {2, 10, 18, 26},
            {3, 11, 19, 27},
            {4, 12, 20, 28},
            {5, 13, 21, 29},
            {6, 14, 22, 30},
            {7, 15, 23, 31},
            {8, 16, 24, 32}});
    CSRMatrix<float> calculatedSolution(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}}));
    CSRMatrix<float> expectedSolution(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

void t3()
{
    CSRMatrix<float> sample_Matrix(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {1, 0, 0, 4},
                {5, 6, 0, 0},
                {0, 0, 0, 0},
                {13, 14, 15, 16},
                {0, 18, 19, 0}}));
    DenseMatrix<float> matrixA(
        std::vector<std::vector<float>>{
            {1, 2, 3, 4, 5, 6, 7, 8},
            {9, 10, 11, 12, 13, 14, 15, 16},
            {17, 18, 19, 20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29, 30, 31, 32},
            {33, 34, 35, 36, 37, 38, 39, 40}});
    DenseMatrix<float> matrixB(
        std::vector<std::vector<float>>{
            {1, 9, 17, 25},
            {2, 10, 18, 26},
            {3, 11, 19, 27},
            {4, 12, 20, 28},
            {5, 13, 21, 29},
            {6, 14, 22, 30},
            {7, 15, 23, 31},
            {8, 16, 24, 32}});
    CSRMatrix<float> calculatedSolution(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}}));
    CSRMatrix<float> expectedSolution(
        DenseMatrix(
            std::vector<std::vector<float>>{
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 0, 0, 0}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

int main()
{
    printf("Running tests...\n");
    // t1();
    // t2();
    t3();

    // TODO: more tests!
    return 0;
}