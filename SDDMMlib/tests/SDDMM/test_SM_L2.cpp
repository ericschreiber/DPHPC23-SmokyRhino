#include <cassert>
#include <iostream>
#include <vector>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SM_L2/SM_L2_GPU.hpp"

void run_testcase(COOMatrix<float> sample_Matrix, DenseMatrix<float> matrixA, DenseMatrix<float> matrixB, COOMatrix<float> calculatedSolution, COOMatrix<float> expectedSolution)
{
    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    sm_l2_SDDMM_GPU<float>* class_to_run = new sm_l2_SDDMM_GPU<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        1,
        std::bind(
            &sm_l2_SDDMM_GPU<float>::SDDMM,
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
        std::cout << std::endl;
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
        std::cout << std::endl;
    }

    return;
}

void t1()
{
    COOMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    COOMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    COOMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 0, 0}, {0, 0, 0}, {0, 0, 0}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

int main()
{
    printf("Running tests...\n");
    t1();

    // TODO: more tests!
    return 0;
}