// create matrix, write to file, read and check for equality
#include "CSRMatrix.hpp"
#include <stdio.h>

int main(){
    CSRMatrix<double> matrixDouble(3, 3);
    CSRMatrix<int> matrixInt(3, 3);

    //Initialize matrixDouble
    matrixDouble.values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrixDouble.colIndices = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    matrixDouble.rowPtr = {0, 3, 6, 9};
    matrixDouble.numRows = 3;
    matrixDouble.numCols = 3;

    //Initialize matrixInt
    matrixInt.values = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matrixInt.colIndices = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    matrixInt.rowPtr = {0, 3, 6, 9};
    matrixInt.numRows = 3;
    matrixInt.numCols = 3;

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
    if (matrixDoubleFromFile.values == matrixDouble.values && 
        matrixDoubleFromFile.colIndices == matrixDouble.colIndices && 
        matrixDoubleFromFile.rowPtr == matrixDouble.rowPtr && 
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
        matrixIntFromFile.colIndices == matrixInt.colIndices && 
        matrixIntFromFile.rowPtr == matrixInt.rowPtr && 
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

    // remove the temporary files
    remove("testDouble.txt");
    remove("testInt.txt");

    return 0;
}