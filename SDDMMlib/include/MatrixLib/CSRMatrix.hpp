// CSRMatrix.hpp
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <string>
#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class CSRMatrix: virtual public SparseMatrix<T> {
public:
    CSRMatrix(int rows, int cols);  // Constructor for an empty CSR matrix
    CSRMatrix(const std::vector<std::vector<T>>& values);  // Constructor from dense matrix
    CSRMatrix(const CSRMatrix& other);  // Copy constructor
    virtual void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, SparseMatrix<T>& result,
       std::function<void(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result)> SDDMMFunc) const override;
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

    virtual bool operator == (const SparseMatrix<T>& other) const override;

    virtual int getNumRows() const override;
    virtual int getNumCols() const override;
    virtual T at(int row, int col) const override;
    virtual int getNumValues() const override;
    virtual std::vector<T> getValues() const override;
    virtual std::vector<int> getColIndices() const override;
    virtual std::vector<int> getRowPtr() const override;


private:
    std::vector<T> values;
    std::vector<int> colIndices;
    std::vector<int> rowPtr;
    int numRows;
    int numCols;
};

#endif // CSR_MATRIX_HPP
