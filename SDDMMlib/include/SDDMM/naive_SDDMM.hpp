// naive_SDDMM.hpp
#ifndef NAIVE_SDDMM_HPP
#define NAIVE_SDDMM_HPP

#include "SDDMM/SDDMMlib.hpp"
#include "MatrixLib/SparseMatrix.hpp"
#include "MatrixLib/DenseMatrix.hpp"

template <typename T>
class naive_SDDMM : public SDDMMlib<T> {
    public:
        static void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result);
};

#endif // NAIVE_SDDMM_HPP