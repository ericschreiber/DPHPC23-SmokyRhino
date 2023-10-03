// SparseMatrix.hpp
#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <vector>
#include <string>

template <typename T>
class SparseMatrix {
public:
    virtual void SDDMM(const denseMatrix& x, const denseMatrix& y, const SparseMatrix& z, SparseMatrix& result
        void (*SDDMMFunc)(const denseMatrix& x, const denseMatrix& y, const SparseMatrix& z, SparseMatrix& result)) const = 0;
    virtual void readFromFile(const std::string& filePath) = 0;
    virtual void writeToFile(const std::string& filePath) const = 0;
    virtual ~SparseMatrix() {}
};

#endif // SPARSE_MATRIX_HPP