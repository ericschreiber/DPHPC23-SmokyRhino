// CSRMatrix.cpp
#include "CSRMatrix.hpp"
#include <fstream>
#include <iostream>
#include <cassert>

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
    file >> dataType;


    // Resize the CSR matrix data structures
    values.resize(numNonZeros);
    colIndices.resize(numNonZeros);
    rowPtr.resize(numRows + 1);

    // Read the values, column indices, and row pointers from the file
    for (int i = 0; i < numNonZeros; ++i) {
        file >> values[i] >> colIndices[i];
    }

    for (int i = 0; i <= numRows; ++i) {
        file >> rowPtr[i];
    }

    file.close();
}

template <typename T>   
void CSRMatrix<T>::SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, SparseMatrix<T>& result,
    std::function<void(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result)> SDDMMFunc) const {
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

// OPerator overloading ==
template <typename T>  
bool CSRMatrix<T>::operator == (const SparseMatrix<T>& other) const {
    // Check if the dimensions are the same
    if (numRows != other.getNumCols() || numCols != other.getNumCols()) {
        std::cout << "Error: Dimensions are not the same" << std::endl;
        return false;
    }

    // Check if the values are the same
    if (values != other.getValues()) {
        std::cout << "Error: Values are not the same" << std::endl;
        for (int i = 0; i < values.size(); ++i) {
            std::cout << values[i] << " " << other.getValues()[i] << std::endl;
        }
        return false;
    }

    // Check if the column indices are the same
    if (colIndices != other.getColIndices()) {
        std::cout << "Error: Column indices are not the same" << std::endl;
        return false;
    }

    // Check if the row pointers are the same
    if (rowPtr != other.getRowPtr()) {
        std::cout << "Error: Row pointers are not the same" << std::endl;
        return false;
    }

    return true;
}


template <typename T>
int CSRMatrix<T>::getNumRows() const {
    return numRows;
}

template <typename T>
int CSRMatrix<T>::getNumCols() const {
    return numCols;
}

template <typename T>
T CSRMatrix<T>::at(int row, int col) const {
    // Assert that the row and column are in bounds
    assert(row >= 0 && row < numRows && "Error: Row index out of bounds");
    assert(col >= 0 && col < numCols && "Error: Column index out of bounds");

    // Find the index of the value in the row
    int index = -1;
    for (int i = rowPtr[row]; i < rowPtr[row + 1]; ++i) {
        if (colIndices[i] == col) {
            index = i;
            break;
        }
    }

    // Return the value if it was found, otherwise return -1
    assert(index != -1 && "Error: Value not found");
    return values[index];
    
}

template <typename T>
std::vector<T> CSRMatrix<T>::getValues() const {
    return values;
}

template <typename T>
int CSRMatrix<T>::getNumValues() const {
    return values.size();
}

template <typename T>
std::vector<int> CSRMatrix<T>::getColIndices() const {
    return colIndices;
}

template <typename T>
std::vector<int> CSRMatrix<T>::getRowPtr() const {
    return rowPtr;
}



// Instantiate the versions of the class that we need
template class CSRMatrix<float>;
template class CSRMatrix<double>;
template class CSRMatrix<int>;
