#include "DenseMatrix.hpp"
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cassert>

// Create a dense matrix, write and read from file, and check for equality

int main(){
    DenseMatrix<double> matrixDouble(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    DenseMatrix<int> matrixInt(std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

    // Write matrixDouble to file
    matrixDouble.writeToFile("testDouble.txt");

    // Write matrixInt to file
    matrixInt.writeToFile("testInt.txt");

    // Read matrixDouble from file
    DenseMatrix<double> matrixDoubleFromFile(3, 3);
    matrixDoubleFromFile.readFromFile("testDouble.txt");

    // Read matrixInt from file
    DenseMatrix<int> matrixIntFromFile(3, 3);
    matrixIntFromFile.readFromFile("testInt.txt");

    // Check if matrixDoubleFromFile is equal to matrixDouble
    bool same = true;
    for (int i = 0; i < matrixDoubleFromFile.getNumRows(); ++i) {
        for (int j = 0; j < matrixDoubleFromFile.getNumCols(); ++j) {
            if (matrixDoubleFromFile.getValue(i, j) != matrixDouble.getValue(i, j)) {
                same = false;
            }
        }
    }
    assert(same && "matrixDoubleFromFile is not equal to matrixDouble");

    // Check if matrixIntFromFile is equal to matrixInt
    same = true;
    for (int i = 0; i < matrixIntFromFile.getNumRows(); ++i) {
        for (int j = 0; j < matrixIntFromFile.getNumCols(); ++j) {
            if (matrixIntFromFile.getValue(i, j) != matrixInt.getValue(i, j)) {
                same = false;
            }
        }
    }
    assert(same && "matrixIntFromFile is not equal to matrixInt");

    std::cout << "Test passed!" << std::endl;

    // remove the temp files
    remove("testDouble.txt");
    remove("testInt.txt");

    return 0;

}