#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_SDDMM.hpp"
#include <iostream>
#include <cassert>
#include <vector>

int main() {
    CSRMatrix<double> matrixA(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    DenseMatrix<double> matrixB(std::vector<std::vector<double>>{{1}, {2}, {3}});
    DenseMatrix<double> matrixC(std::vector<std::vector<double>>{{1}, {2}, {3}});
    CSRMatrix<double> calculatedSolution(std::vector<std::vector<double>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    CSRMatrix<double> expectedSolution(std::vector<std::vector<double>>{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}});

    naive_SDDMM<double> naive_SDDMM;

    // Call multiply and pass the multiplication function from the library
    matrixA.SDDMM(matrixB, matrixC, calculatedSolution, &(naive_SDDMM.SDDMM));

    // Check if the calculated solution is equal to the expected solution
    if (calculatedSolution == expectedSolution) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        auto calculatedValues = calculatedSolution.getValues();
        for (int i = 0; i < calculatedValues.size(); ++i) {
            std::cout << calculatedValues.at(i) << " ";
        }
        
    }

    return 0;
}