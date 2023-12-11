#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_SDDMM.hpp"

int main()
{
    CSRMatrix<float> matrixA(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1}, {2}, {3}});
    DenseMatrix<float> matrixC(std::vector<std::vector<float>>{{1}, {2}, {3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    const int num_iterations = 1;
    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    naive_SDDMM<float>* class_to_run = new naive_SDDMM<float>(&timer);
    // Either use construtor to set timer or use set_timer
    // class_to_run->set_timer(&timer);

    // Call multiply and pass the multiplication function from the library
    matrixA.SDDMM(
        matrixB,
        matrixC,
        calculatedSolution,
        num_iterations,
        std::bind(
            &naive_SDDMM<float>::SDDMM,
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
        std::cout << "SDDMM: all tests passed!" << std::endl;
    }
    else
    {
        std::cout << "SDDMM: Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
    }

    return 0;
}