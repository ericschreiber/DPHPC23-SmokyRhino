#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_SDDMM.hpp"

int main()
{
    CSRMatrix<double> matrixA_HOST(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    DenseMatrix<double> matrixB_HOST(std::vector<std::vector<double>>{{1}, {2}, {3}});
    DenseMatrix<double> matrixC_HOST(std::vector<std::vector<double>>{{1}, {2}, {3}});
    CSRMatrix<double> calculatedSolution_HOST(std::vector<std::vector<double>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    CSRMatrix<double> expectedSolution_HOST(std::vector<std::vector<double>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});

    // Call multiply and pass the multiplication function from the library
    matrixA_HOST.SDDMM(matrixB_HOST, matrixC_HOST, calculatedSolution_HOST, std::bind(&naive_SDDMM<double>::SDDMM, naive_SDDMM<double>(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

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