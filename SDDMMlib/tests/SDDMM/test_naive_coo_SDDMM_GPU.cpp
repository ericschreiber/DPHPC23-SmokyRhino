#include <cassert>
#include <iostream>
#include <vector>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_coo_gpu/naive_coo_SDDMM_GPU.hpp"

std::tuple<COOMatrix<float>*, DenseMatrix<float>*, DenseMatrix<float>*, COOMatrix<float>*, COOMatrix<float>*> get_simple_near_zeros()
{
    COOMatrix<float>* matrixA = new COOMatrix<float>(DenseMatrix(std::vector<std::vector<float>>{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    DenseMatrix<float>* matrixB = new DenseMatrix<float>(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float>* matrixC = new DenseMatrix<float>(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    COOMatrix<float>* calculatedSolution = new COOMatrix<float>(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    COOMatrix<float>* expectedSolution = new COOMatrix<float>(DenseMatrix(std::vector<std::vector<float>>{{2, 0, 0}, {0, 0, 0}, {0, 0, 0}}));

    return std::make_tuple(matrixA, matrixB, matrixC, calculatedSolution, expectedSolution);
}

std::tuple<COOMatrix<float>*, DenseMatrix<float>*, DenseMatrix<float>*, COOMatrix<float>*, COOMatrix<float>*> get_complex_small()
{
    COOMatrix<float>* matrixA = new COOMatrix<float>(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
    DenseMatrix<float>* matrixB = new DenseMatrix<float>(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float>* matrixC = new DenseMatrix<float>(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    COOMatrix<float>* calculatedSolution = new COOMatrix<float>(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    COOMatrix<float>* expectedSolution = new COOMatrix<float>(DenseMatrix(std::vector<std::vector<float>>{{2, 4, 6}, {4, 8, 12}, {6, 12, 18}}));
    return std::make_tuple(matrixA, matrixB, matrixC, calculatedSolution, expectedSolution);
}

// make a function that gets a function which creates the matrices and then runs the test
void test(std::function<std::tuple<COOMatrix<float>*, DenseMatrix<float>*, DenseMatrix<float>*, COOMatrix<float>*, COOMatrix<float>*>()> getMatrices)
{
    auto matrices = getMatrices();
    COOMatrix<float>* matrixA = std::get<0>(matrices);
    DenseMatrix<float>* matrixB = std::get<1>(matrices);
    DenseMatrix<float>* matrixC = std::get<2>(matrices);
    COOMatrix<float>* calculatedSolution = std::get<3>(matrices);
    COOMatrix<float>* expectedSolution = std::get<4>(matrices);

    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    naive_coo_SDDMM_GPU<float>* class_to_run = new naive_coo_SDDMM_GPU<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    matrixA->SDDMM(
        *matrixB,
        *matrixC,
        *calculatedSolution,
        std::bind(
            &naive_coo_SDDMM_GPU<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4));

    delete class_to_run;
    class_to_run = nullptr;

    // Check if the calculated solution is equal to the expected solution
    if (*calculatedSolution == *expectedSolution)
    {
        std::cout << "Test passed!" << std::endl;
    }
    else
    {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution->getValues();
        for (int i = 0; i < calculatedValues.size(); ++i)
        {
            std::cout << calculatedValues.at(i) << " ";
        }
        std::cout << std::endl;
    }

    // print the result
    std::cout << "Result: " << std::endl;
    for (int i = 0; i < calculatedSolution->getNumRows(); i++)
    {
        for (int j = 0; j < calculatedSolution->getNumCols(); j++)
        {
            std::cout << calculatedSolution->at(i, j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    delete matrixA;
    matrixA = nullptr;
    delete matrixB;
    matrixB = nullptr;
    delete matrixC;
    matrixC = nullptr;
    delete calculatedSolution;
    calculatedSolution = nullptr;
    delete expectedSolution;
    expectedSolution = nullptr;

    return;
}

int main()
{
    test(get_simple_near_zeros);
    test(get_complex_small);
    return 0;
}