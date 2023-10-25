// DenseMatrix.hpp
#ifndef DENSEMATRIX_HPP
#define DENSEMATRIX_HPP

#include <string>
#include <vector>

// if I include "CRSMatrix.hpp" here, I get a circular dependency error
// when I build the project therefore I am now trying forward declaration
template <typename T>
class CSRMatrix;

template <typename T>
class DenseMatrix
{
    public:
        // Constructors
        DenseMatrix(int rows, int cols);                         // Constructor for an empty dense matrix
        DenseMatrix(const std::vector<std::vector<T>>& values);  // Copy constructor
        DenseMatrix(CSRMatrix<T>& csrMatrix);                    // constructor to convert CSR matrix to dense matrix

        int getNumRows() const;
        int getNumCols() const;
        std::vector<std::vector<T>> getValues();  // added this, don't see why we should not have it
        T at(int row, int col) const;
        void setValue(int row, int col, T value);
        void transpose();

        void readFromFile(const std::string& filePath);
        void writeToFile(const std::string& filePath) const;

    private:
        std::vector<std::vector<T>> values;
        int numRows;
        int numCols;
};

#endif  // DENSEMATRIX_HPP