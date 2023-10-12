// DenseMatrix.cpp
#include "DenseMatrix.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>

template <typename T>
DenseMatrix<T>::DenseMatrix(int rows, int cols) : numRows(rows), numCols(cols), values(rows, std::vector<T>(cols, T())) {}

template <typename T>
DenseMatrix<T>::DenseMatrix(const std::vector<std::vector<T>>& values) : numRows(values.size()), numCols(values[0].size()), values(values) {}

template <typename T>
int DenseMatrix<T>::getNumRows() const {
    return numRows;
}

template <typename T>
int DenseMatrix<T>::getNumCols() const {
    return numCols;
}

template <typename T>
T DenseMatrix<T>::at(int row, int col) const {
   // check out of range
   if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
       throw std::out_of_range("Error: DenseMatrix::at() out of range");
   }
   else {
       return values[row][col];
   }
}

template <typename T>
void DenseMatrix<T>::setValue(int row, int col, T value) {
   // check out of range
   if (row < 0 || row >= numRows || col < 0 || col >= numCols) {
       throw std::out_of_range("Error: DenseMatrix::setValue() out of range");
   }
   else {
       values[row][col] = value;
   }
}

template <typename T>
void DenseMatrix<T>::readFromFile(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filePath << std::endl;
        return;
    }
    
    std::string datatype;

    // Read numRows, numCols, datatype
    file >> numRows >> numCols >> datatype;
    values.resize(numRows, std::vector<T>(numCols, T()));

    // Check the datatype
    if (datatype != typeid(T).name()) {
        std::cerr << "Error: Datatype in file does not match the datatype of the matrix" << std::endl;
        return;
    }

    // Read the values
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; j++) {
            if (!(file >> values[i][j])) {
                std::cerr << "Error: Could not read value from file even though more should be there." << std::endl;
                return;
            }
        }
    }

    file.close();
}

template <typename T>
void DenseMatrix<T>::writeToFile(const std::string& filePath) const {
    // The file format is:
    //     numRows numCols datatype
    //     value value value ...
    std::ofstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filePath << std::endl;
        return;
    }

    // Write numRows, numCols, datatype
    file << numRows << " " << numCols << " " << typeid(T).name() << std::endl;

    // Write the values
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; j++) {
            file << values[i][j] << " ";
        }
        file << std::endl;
    }

    file.close();
}

template <typename T>
T DenseMatrix<T>::operator[](int row, int col) const {
    return values[row][col];
}


// Instanciate the template class with all the types we want to support
template class DenseMatrix<int>;
template class DenseMatrix<float>;
template class DenseMatrix<double>;
