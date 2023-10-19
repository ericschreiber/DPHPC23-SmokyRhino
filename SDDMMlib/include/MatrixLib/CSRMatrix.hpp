// CSRMatrix.hpp
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <string>
#include <vector>

#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class CSRMatrix : virtual public SparseMatrix<T>
{
    public:
        CSRMatrix();                              // Default constructor
        CSRMatrix(int rows, int cols);            // Constructor for an empty CSR matrix
        CSRMatrix(const DenseMatrix<T>& values);  // Constructor from dense matrix
        CSRMatrix(const CSRMatrix& other);        // Copy constructor
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            SparseMatrix<T>& result,
            std::function<void(
                const DenseMatrix<T>& x,
                const DenseMatrix<T>& y,
                const SparseMatrix<T>& z,
                SparseMatrix<T>& result)> SDDMMFunc)
            const override;
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

        virtual bool operator==(const SparseMatrix<T>& other) const override;

        virtual int getNumRows() const override;
        virtual int getNumCols() const override;
        virtual T at(int row, int col) const override;
        virtual int getNumValues() const override;
        virtual std::vector<T> getValues() const override;
        virtual std::vector<int> getColIndices() const override;
        virtual std::vector<int> getRowPtr() const override;

        virtual void setValues(const std::vector<T>& values) override;
        virtual void setColIndices(const std::vector<int>& colIndices) override;
        virtual void setRowPtr(const std::vector<int>& rowPtr) override;

    private:
        std::vector<T> values;
        std::vector<int> colIndices;
        std::vector<int> rowPtr;
        int numRows;
        int numCols;
};

#endif  // CSR_MATRIX_HPP
