// CSRMatrix.hpp
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <string>
#include <vector>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class CSRMatrix : virtual public SparseMatrix<T>
{
    public:
        CSRMatrix();                               // Default constructor
        CSRMatrix(int rows, int cols);             // Constructor for an empty CSR matrix
        CSRMatrix(const DenseMatrix<T>& values);   // Constructor from dense matrix
        CSRMatrix(const COOMatrix<T>& cooMatrix);  // Constructor from COO matrix
        CSRMatrix(const CSRMatrix& other);         // Copy constructor
        // CSRMatrix(const SparseMatrix<T>& other);   // Copy constructor
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            SparseMatrix<T>& result,
            const int num_iterations,
            std::function<void(
                const DenseMatrix<T>& x,
                const DenseMatrix<T>& y,
                const SparseMatrix<T>& z,
                SparseMatrix<T>& result,
                const int num_iterations)> SDDMMFunc)
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

        virtual int getNumRows() const override;
        virtual int getNumCols() const override;
        virtual T at(int row, int col) const override;
        virtual int getNumValues() const override;
        virtual const std::vector<T>& getValues() const override;
        virtual const std::vector<int>& getColIndices() const override;
        virtual const std::vector<int>& getRowArray() const override;

        virtual void setValues(const std::vector<T>& values) override;
        virtual void setColIndices(const std::vector<int>& colIndices) override;
        virtual void setRowArray(const std::vector<int>& rowPtr) override;

    private:
        virtual bool isEqual(const SparseMatrix<T>& other) const override;
        std::vector<T> values;
        std::vector<int> colIndices;
        std::vector<int> rowPtr;
        int numRows;
        int numCols;
};

#endif  // CSR_MATRIX_HPP