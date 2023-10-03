// naive_SDDMM.hpp
#ifndef NAIVE_SDDMM_HPP
#define NAIVE_SDDMM_HPP

#include "SDDMMlib.hpp"
#include "CSRMatrix.hpp"

class naive_SDDMM : public SDDMMlib {
    public:
        static void SDDMM(const denseMatrix& x, const denseMatrix& y, const SparseMatrix& z, SparseMatrix& result);
};