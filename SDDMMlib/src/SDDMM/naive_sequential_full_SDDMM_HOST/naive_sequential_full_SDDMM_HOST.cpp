// naive_sequential_full_SDDMM.cpp
#include "naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.hpp"

#include <iostream>

#include "DenseMatrix.hpp"

// Only the float type of the class is valid all other types will throw an error
void naive_sequential_full_SDDMM_HOST<float>::SDDMM(
    const DenseMatrix<float>& x,
    const DenseMatrix<float>& y,
    const SparseMatrix<float>& z,
    SparseMatrix<float>& result) const
{
    // Check if CSRMatrix
    const CSRMatrix<float>* csrMatrix = dynamic_cast<const CSRMatrix<float>*>(&z);
    CSRMatrix<float>* csrResult = dynamic_cast<CSRMatrix<float>*>(&result);
    if (csrMatrix == nullptr || csrResult == nullptr)
    {
        throw std::invalid_argument("Error: naive_sequential_full_SDDMM_HOST::SDDMM() only accepts CSRMatrix<float> as input. Other formats are not supported yet");
    }
    else
    {
        naive_sequential_full_SDDMM_HOST_CSR(x, y, *csrMatrix, *csrResult);
    }

    csrMatrix = nullptr;
    csrResult = nullptr;

    return;
}

template <typename T>
void naive_sequential_full_SDDMM_HOST<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    assert(false && "Error: naive_sequential_full_SDDMM_HOST::SDDMM() only accepts float as input. Other types are not supported yet");
}

void naive_sequential_full_SDDMM_HOST<float>::naive_sequential_full_SDDMM_HOST_CSR(
    const DenseMatrix<float>& x,
    const DenseMatrix<float>& y,
    const CSRMatrix<float>& z,
    CSRMatrix<float>& result) const
{
    // This is a very dumb implementation, because it samples only AFTER the
    // matrix x matrix multiplication

    int m = x.getNumRows();
    int n = x.getNumCols();
    int k = y.getNumCols();

    // size check
    assert(m == z.getNumRows());
    assert(n == z.getNumCols());
    assert(k == n);

    auto xy = DenseMatrix<float>(m, n);
    std::vector<float> temp_vals(z.getNumValues());

    // one explicitly Please note that the paper uses A[M][K] and B[N][K]. I.e. B
    // already seems to be transposed! I am making the same assumption
    // I also assume we are taking a CRS matrix

    this->start_run();

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int l = 0; l < k; l++)
            {
                auto mul = x.at(i, l) * y.at(j, l);
                auto curr_xy_ij = xy.at(i, j);
                xy.setValue(i, j, mul + curr_xy_ij);
            }
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = z.getRowArray()[i]; j < z.getRowArray()[i + 1]; j++)
        {
            temp_vals[j] = xy.at(i, z.getColIndices()[j]) * z.getValues()[j];
        }
    }

    this->stop_run();

    result.setValues(temp_vals);
    result.setColIndices(z.getColIndices());
    result.setRowArray(z.getRowArray());

    std::cout << "naive_sequential_full_SDDMM was executed :)" << std::endl;
    return;
}

void naive_sequential_full_SDDMM_HOST<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: naive_sequential_full_SDDMM_HOST::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_cpu_run();  // OR _timer.start_gpu_run();
}

void naive_sequential_full_SDDMM_HOST<float>::stop_run() const
{
    this->_timer->stop_cpu_run();  // OR _timer.stop_gpu_run();
}

// Explicit template instantiation
// template class naive_sequential_full_SDDMM_HOST<float>;
template class naive_sequential_full_SDDMM_HOST<double>;
template class naive_sequential_full_SDDMM_HOST<int>;