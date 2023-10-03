// SparseMatrix.hpp
#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>
#include <string>
#include "MatrixLib/DenseMatrix.hpp"

template <typename T>
class SparseMatrix {
public:
    virtual void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix& z, SparseMatrix& result, void (*SDDMMFunc)(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix& z, SparseMatrix& result)) const = 0;
    virtual void readFromFile(const std::string& filePath) = 0;
    virtual void writeToFile(const std::string& filePath) const = 0;
    virtual ~SparseMatrix() {}
};

#endif // SPARSE_MATRIX_HPP