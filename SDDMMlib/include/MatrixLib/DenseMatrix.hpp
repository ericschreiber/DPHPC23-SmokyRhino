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
class SparseMatrix;

template <typename T>
class DenseMatrix
{
    public:
        // Constructors
        DenseMatrix(int rows, int cols);                         // Constructor for an empty dense matrix
        DenseMatrix(const std::vector<std::vector<T>>& values);  // Help for using the constructor
        DenseMatrix(const SparseMatrix<T>& sparseMatrix);        // constructor to convert sparse matrix to dense matrix
        DenseMatrix(const DenseMatrix<T>& denseMatrix);          // Copy constructor
        ~DenseMatrix();

        int getNumRows() const;
        int getNumCols() const;
        const T* getValues() const;  // added this, don't see why we should not have it
        T at(int row, int col) const;
        void setValue(int row, int col, T value);
        void transpose();
        void setValues(const T* values, int size);

        void readFromFile(const std::string& filePath);
        void writeToFile(const std::string& filePath) const;

    private:
        void convert_csr_dense(const CSRMatrix<T>& csrMatrix);  // constructor to convert CSR matrix to dense matrix
        T* values = nullptr;
        int numRows = 0;
        int numCols = 0;
};

#endif  // DENSEMATRIX_HPP