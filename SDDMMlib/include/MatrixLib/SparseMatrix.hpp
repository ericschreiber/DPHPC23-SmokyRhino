// SparseMatrix.hpp
#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <functional>
#include <string>
#include <vector>

#include "DenseMatrix.hpp"

template <typename T>
class SparseMatrix
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            SparseMatrix<T>& result,
            std::function<void(
                const DenseMatrix<T>& x,
                const DenseMatrix<T>& y,
                const SparseMatrix<T>& z,
                SparseMatrix<T>& result)> SDDMMFunc) const = 0;
        virtual void readFromFile(const std::string& filePath) = 0;
        virtual void writeToFile(const std::string& filePath) const = 0;
        virtual ~SparseMatrix() {}
        bool operator==(const SparseMatrix& other) const { return isEqual(other); }
        virtual int getNumRows() const = 0;
        virtual int getNumCols() const = 0;
        virtual T at(int row, int col) const = 0;
        virtual int getNumValues() const = 0;
        virtual const std::vector<T>& getValues() const = 0;
        virtual const std::vector<int>& getColIndices() const = 0;
        virtual const std::vector<int>& getRowArray() const = 0;  // a "rowArray" can either be the rowPointer vector (CSR) or a vector of rowIndices (COO).
                                                                  // Which one getRowArray returns depends on the class that implements the SparseMatrix interface.

        virtual void setValues(const std::vector<T>& values) = 0;
        virtual void setColIndices(const std::vector<int>& colIndices) = 0;
        virtual void setRowArray(const std::vector<int>& rowPtr) = 0;

    private:
        virtual bool isEqual(const SparseMatrix& other) const = 0;
};

#endif  // SPARSE_MATRIX_HPP