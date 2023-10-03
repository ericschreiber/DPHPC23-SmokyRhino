// SDDMMlib.hpp
// #ifndef SDDMMLIB_HPP
// #define SDDMMLIB_HPP

#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"

template <typename T>
class SDDMMlib {
    public:
        static void SDDMM(const denseMatrix& x, const denseMatrix& y, const SparseMatrix& z, SparseMatrix& result) = 0;
};

#endif // SDDMMLIB_HPP