#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "cusparse_baseline/cusparse_baseline.hpp"

void run_testcase(CSRMatrix<float> sample_Matrix, DenseMatrix<float> matrixA, DenseMatrix<float> matrixB, CSRMatrix<float> calculatedSolution, CSRMatrix<float> expectedSolution)
{
    const int num_iterations = 10;

    // Set a timer
    ExecutionTimer timer = ExecutionTimer();
    cusparse_baseline<float>* class_to_run = new cusparse_baseline<float>(&timer);

    // Call multiply and pass the multiplication function from the library
    sample_Matrix.SDDMM(
        matrixA,
        matrixB,
        calculatedSolution,
        num_iterations,
        std::bind(
            &cusparse_baseline<float>::SDDMM,
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
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{2, 0, 0}, {0, 0, 0}, {0, 0, 0}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 0, 0}, {0, 0, 0}, {0, 0, 0}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

void t2()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 0}, {2, 0}, {3, 0}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

void t3()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1}, {2, 2}, {3, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3}, {1, 2, 3}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}, {1, 2, 3}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{2, 4, 6}, {4, 8, 12}, {6, 12, 18}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

// checks if the code works when last tile of a row is smaller than shared memory.
// requires SHARED_MEM_SIZE_BYTES to be set to 8.
void t4()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{
        1,
        1,
    }}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 2, 3}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 1}, {1, 1}, {1, 1}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{6, 6}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{6, 6}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

void t5_12x12()
{
    CSRMatrix<float> sample_Matrix(DenseMatrix(std::vector<std::vector<float>>{{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}));
    DenseMatrix<float> matrixA(std::vector<std::vector<float>>{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}, {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5}, {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6}, {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7}, {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8}, {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9}, {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}, {11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11}, {12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12}});
    DenseMatrix<float> matrixB(std::vector<std::vector<float>>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}});
    CSRMatrix<float> calculatedSolution(DenseMatrix(std::vector<std::vector<float>>{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}));
    CSRMatrix<float> expectedSolution(DenseMatrix(std::vector<std::vector<float>>{{12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1200, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}));

    run_testcase(sample_Matrix, matrixA, matrixB, calculatedSolution, expectedSolution);
}

int main()
{
    t1();
    t2();
    t3();
    t4();
    t5_12x12();
    // also ran all the test functions with SHARED_MEM_SIZE_BYTES = 4 which forces tiling to happen (and it worked)
    // also ran with THREADS_PER_BLOCK = 1 in which case the spawned threads have more than one float to work on (and it worked)

    // TODO: more tests!
    return 0;
}