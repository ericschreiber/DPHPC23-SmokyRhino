// create matrix, write to file, read and check for equality
#include <stdio.h>

#include <cassert>
#include <iostream>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"

int main()
{
    CSRMatrix<double> matrixDouble(DenseMatrix(std::vector<std::vector<double>>{{1.1, 2.2, 3.3}, {4.4, 5.5, 6.6}, {7.7, 8.8, 9.9}}));
    CSRMatrix<int> matrixInt(DenseMatrix(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}));

    // Write matrixDouble to file
    matrixDouble.writeToFile("testDouble.txt");

    // Write matrixInt to file
    matrixInt.writeToFile("testInt.txt");

    // Read matrixDouble from file
    CSRMatrix<double> matrixDoubleFromFile(3, 3);
    matrixDoubleFromFile.readFromFile("testDouble.txt");

    // Read matrixInt from file
    CSRMatrix<int> matrixIntFromFile(3, 3);
    matrixIntFromFile.readFromFile("testInt.txt");

    // Check if matrixDoubleFromFile is equal to matrixDouble
    assert(matrixDoubleFromFile == matrixDouble && "matrixDoubleFromFile is not equal to matrixDouble");

    // Check if matrixIntFromFile is equal to matrixInt
    assert(matrixInt == matrixIntFromFile && "matrixIntFromFile is not equal to matrixInt");

    std::cout << "IO CSR: all tests passed!" << std::endl;

    // remove the temporary files
    remove("testDouble.txt");
    remove("testInt.txt");

    return 0;
}