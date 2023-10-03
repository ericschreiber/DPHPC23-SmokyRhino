// CSRMatrix.hpp
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include "SparseMatrix.hpp"
#include <vector>
#include <string>
#include "SDDMMlib.hpp"

template <typename T>
class CSRMatrix : public SparseMatrix {
public:
    CSRMatrix(int rows, int cols);  // Constructor for an empty CSR matrix
    CSRMatrix(const std::vector<std::vector<T>>& values);  // Constructor from dense matrix
    CSRMatrix(const CSRMatrix& other);  // Copy constructor
    virtual void SDDMM(const denseMatrix& x, const denseMatrix& y, SparseMatrix& result
        void (*SDDMMFunc)(const denseMatrix& x, const denseMatrix& y, const SparseMatrix& z, SparseMatrix& result)) const override;
    // Read CSR matrix from a file where a matrix was stored in CSR format using writeToFile()
    virtual void readFromFile(const std::string& filePath) override; 
    // Write CSR matrix to a file in the following format:
    //     numRows numCols
    //     numNonZeros
    //     datatype
    //     value colIndex
    //     ...
    //     row pointers   
    virtual void writeToFile(const std::string& filePath) const override;
    virtual ~CSRMatrix() {}

private:
    std::vector<T> values;
    std::vector<int> colIndices;
    std::vector<int> rowPtr;
    int numRows;
    int numCols;
};

#endif // CSR_MATRIX_HPP
