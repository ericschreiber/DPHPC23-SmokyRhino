// CSRMatrix.cpp
#include "CSRMatrix.hpp"
#include <fstream>
#include <iostream>

template <typename T>
CSRMatrix<T>::CSRMatrix(int rows, int cols) : numRows(rows), numCols(cols) {
    // Initialize CSRMatrix with zeros
    values.clear();
    colIndices.clear();
    rowPtr.resize(rows + 1, 0);
}

template <typename T>
CSRMatrix<T>::CSRMatrix(const std::vector<std::vector<T>>& denseMatrix) {
    // Convert a dense matrix to CSR format
    numRows = denseMatrix.size();
    numCols = denseMatrix[0].size();


    // Resize the CSR matrix data structures
    values.clear();
    colIndices.clear();
    rowPtr.resize(numRows + 1, 0);

    // Iterate over the dense matrix and add non-zero values to the CSR matrix
    for (int i = 0; i < numRows; ++i) {
        rowPtr[i] = values.size();
        for (int j = 0; j < numCols; ++j) {
            if (denseMatrix[i][j] != 0.0) {
                values.push_back(denseMatrix[i][j]);
                colIndices.push_back(j);
            }
        }
    }
    rowPtr[numRows] = values.size();
}

template <typename T>
CSRMatrix<T>::CSRMatrix(const CSRMatrix& other) {
    // Copy constructor
    numRows = other.numRows;
    numCols = other.numCols;
    values = other.values;
    colIndices = other.colIndices;
    rowPtr = other.rowPtr;
}

template <typename T>
void CSRMatrix<T>::readFromFile(const std::string& filePath) {
    // Read CSR matrix from a file where a matrix was stored in CSR format using writeToFile()
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filePath << std::endl;
        return;
    }

    // Read numRows, numCols, and the number of non-zero values from the file and the datatype
    file >> numRows >> numCols;
    int numNonZeros;
    file >> numNonZeros;
    std::string dataType;

    // Resize the CSR matrix data structures
    values.resize(numNonZeros);
    colIndices.resize(numNonZeros);
    rowPtr.resize(numRows + 1);

    // Read the values, column indices, and row pointers from the file
    for (int i = 0; i < numNonZeros; ++i) {
        file >> values[i];
        file >> colIndices[i];
    }

    for (int i = 0; i <= numRows; ++i) {
        file >> rowPtr[i];
    }

    file.close();
}

template <typename T>   
void CSRMatrix<T>::SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, SparseMatrix<T>& result,
    void (*SDDMMFunc)(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result)) const {
    // Call the SDDMM function from the library
    SDDMMFunc(x, y, *this, result);
}

template <typename T>
void CSRMatrix<T>::writeToFile(const std::string& filePath) const {
    // Write CSR matrix to a file in the following format:
    //     numRows numCols
    //     numNonZeros
    //     datatype
    //     value colIndex
    //     ...
    //     row pointers

    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filePath << std::endl;
        return;
    }

    // Write numRows, numCols, and the number of non-zero values to the file and the datatype
    file << numRows << " " << numCols << std::endl;
    file << values.size() << std::endl;
    file << typeid(values[0]).name() << std::endl;

    // Write the values and column indices
    for (size_t i = 0; i < values.size(); ++i) {
        file << values[i] << " " << colIndices[i] << std::endl;
    }

    // Write the row pointers
    for (int i = 0; i <= numRows; ++i) {
        file << rowPtr[i] << std::endl;
    }

    file.close();
}
