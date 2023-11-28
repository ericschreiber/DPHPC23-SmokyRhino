#include <cassert>
#include <iostream>
#include <vector>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "tiled_tiles/tiled_tiles.hpp"

void run_testcase(COOMatrix<float> sample_Matrix, DenseMatrix<float> matrixA, DenseMatrix<float> matrixB, COOMatrix<float> calculatedSolution, COOMatrix<float> expectedSolution)
{
    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    tiled_tiles<float>* class_to_run = new tiled_tiles<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        std::bind(
            &tiled_tiles<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4));

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

void t2()
{
    COOMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 0}, {2, 0}, {3, 0}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    COOMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    COOMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

void t3()
{
    COOMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    COOMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    COOMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 4, 6}, {4, 8, 12}, {6, 12, 18}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

// checks if the code works when last tile of a row is smaller than shared memory.
// requires SHARED_MEM_SIZE_BYTES to be set to 8.
void t4()
{
    COOMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{
        1,
        1,
    }}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 2, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 1}, {1, 1}, {1, 1}});
    COOMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{6, 6}}));
    COOMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{6, 6}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

int main()
{
    t1();
    t2();
    t3();
    t4();

    // TODO: more tests!
    return 0;
}