#include <stdio.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"

// Create a dense matrix, write and read from file, and check for equality
void mainTest()
{
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
    for (int i = 0; i < matrixDoubleFromFile.getNumRows(); ++i)
    {
        for (int j = 0; j < matrixDoubleFromFile.getNumCols(); ++j)
        {
            if (matrixDoubleFromFile.at(i, j) != matrixDouble.at(i, j))
            {
                same = false;
            }
        }
    }
    assert(same && "matrixDoubleFromFile is not equal to matrixDouble");

    // Check if matrixIntFromFile is equal to matrixInt
    same = true;
    for (int i = 0; i < matrixIntFromFile.getNumRows(); ++i)
    {
        for (int j = 0; j < matrixIntFromFile.getNumCols(); ++j)
        {
            if (matrixIntFromFile.at(i, j) != matrixInt.at(i, j))
            {
                same = false;
            }
        }
    }
    assert(same && "matrixIntFromFile is not equal to matrixInt");

    std::cout << "Test passed!" << std::endl;

    // remove the temp files
    remove("testDouble.txt");
    remove("testInt.txt");
}

// test the constructor that converts a matrix in csr into a dense matrix
void testCsrToDense()
{
    std::vector<std::vector<int>> in = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    CSRMatrix<int> matrixCSR = CSRMatrix<int>(DenseMatrix<int>(in));
    DenseMatrix<int> matrixDense(matrixCSR);
    const std::vector<std::vector<int>>& vals = matrixDense.getValues();
    // compare both value vectors
    assert(vals == in && "Incorrect CSR to Dense conversion");
}

int main()
{
    mainTest();
    testCsrToDense();

    return 0;
}