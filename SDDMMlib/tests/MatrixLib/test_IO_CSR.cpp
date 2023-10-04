// create matrix, write to file, read and check for equality
#include "CSRMatrix.hpp"
#include <stdio.h>
#include <iostream>
#include <cassert>

int main(){
    CSRMatrix<double> matrixDouble(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    CSRMatrix<int> matrixInt(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

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

    // remove the temporary files
    remove("testDouble.txt");
    remove("testInt.txt");

    return 0;
}