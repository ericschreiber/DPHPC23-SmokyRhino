#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "naive_SDDMM.cpp"

int main() {
    CSRMatrix<double> matrixA(3, 3);
    DenseMatrix<double> matrixB(3, 1);
    DenseMatrix<double> matrixC(3, 1);
    CSRMatrix<double> calculatedSolution(3, 3);
    CSRMatrix<double> expectedSolution(3, 3);

    //Initialize matrixA
    matrixA.values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrixA.colIndices = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    matrixA.rowPtr = {0, 3, 6, 9};
    matrixA.numRows = 3;
    matrixA.numCols = 3;

    //Initialize matrixB
    matrixB.values = {1, 2, 3};
    matrixB.numRows = 3;
    matrixB.numCols = 1;
    
    //Initialize matrixC
    matrixC.values = {1, 2, 3};
    matrixC.numRows = 3;
    matrixC.numCols = 1;

    //Initialize expectedSolution (Hopefully I did this right ^^)
    expectedSolution.values = {14, 32, 50, 32, 77, 122, 50, 122, 194};
    expectedSolution.colIndices = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    expectedSolution.rowPtr = {0, 3, 6, 9};
    expectedSolution.numRows = 3;
    expectedSolution.numCols = 3;

    // Call multiply and pass the multiplication function from the library
    matrixA.SDDMM(matrixB, matrixC, calculatedSolution, &naive_SDDMM::SDDMM);

    // Check if the calculated solution is equal to the expected solution
    if (calculatedSolution == expectedSolution) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed! Calculated solution is: " << std::endl;
        for (int i = 0; i < calculatedSolution.values.size(); ++i) {
            std::cout << calculatedSolution.values[i] << " ";
        }
        
    }

    return 0;
}