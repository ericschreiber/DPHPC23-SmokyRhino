#include "DenseMatrix.hpp"
#include <stdio.h>

// Create a dense matrix, write and read from file, and check for equality

int main(){
    DenseMatrix<double> matrixDouble(3, 3);
    DenseMatrix<int> matrixInt(3, 3);

    //Initialize matrixDouble
    matrixDouble.values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrixDouble.numRows = 3;
    matrixDouble.numCols = 3;

    //Initialize matrixInt
    matrixInt.values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrixInt.numRows = 3;
    matrixInt.numCols = 3;

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
    if (matrixDoubleFromFile.values == matrixDouble.values && 
        matrixDoubleFromFile.numRows == matrixDouble.numRows && 
        matrixDoubleFromFile.numCols == matrixDouble.numCols) {
        std::cout << "Test passed for Doubles!" << std::endl;
    } else {
        std::cout << "Test failed! matrixDoubleFromFile is: " << std::endl;
        for (int i = 0; i < matrixDoubleFromFile.values.size(); ++i) {
            std::cout << matrixDoubleFromFile.values[i] << " ";
        }
        std::cout << std::endl;
    }

    // Check if matrixIntFromFile is equal to matrixInt
    if (matrixIntFromFile.values == matrixInt.values && 
        matrixIntFromFile.numRows == matrixInt.numRows && 
        matrixIntFromFile.numCols == matrixInt.numCols) {
        std::cout << "Test passed for Integers!" << std::endl;
    } else {
        std::cout << "Test failed! matrixIntFromFile is: " << std::endl;
        for (int i = 0; i < matrixIntFromFile.values.size(); ++i) {
            std::cout << matrixIntFromFile.values[i] << " ";
        }
        std::cout << std::endl;
    }

    // remove the temp files
    remove("testDouble.txt");
    remove("testInt.txt");

    return 0;

}