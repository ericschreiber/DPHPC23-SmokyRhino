# SDDMM Libarary
This library contains implementations for matrices and the sddmm algorithms. For that we create two libraries called `SDDMMlib` and `Matrixlib`. The `SDDMMlib` contains the implementations of the algorithms and the `Matrixlib` contains the implementations of the matrices. Both libraries depend on each other (not so nice. TODO: If you want to detangle them give it a go :D). To build it go to `/SDDMMlib/build` and run 
```bash
cmake -DUSE_CUDA=<True/False> ..
cmake --build .
```

## MatrixLib
We have two kinds of matrices: sparse and dense. The sparse matrices is a virtual class to which we can implement other classes. For now there is one for CSR matrices implemented. To implement other compression types please make new classes analogous.
The basic (non extensive) interface is as follows:
```cpp
CSRMatrix(int rows, int cols)  // Constructor for an empty CSR matrix
CSRMatrix(const std::vector<std::vector<T>>& values)  // Constructor from dense matrix

void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, SparseMatrix<T>& result,
       std::function<void(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result)> SDDMMFunc) 
    // Read CSR matrix from a file where a matrix was stored in CSR format using writeToFile()
void readFromFile(const std::string& filePath) 
    // Write CSR matrix to a file in the following format:
    //     numRows numCols
    //     numNonZeros
    //     datatype
    //     value colIndex
    //     ...
    //     row pointers   
void writeToFile(const std::string& filePath) 
```
There are more. Just look into the header file. (/include/MatrixLib/SparseMatrix.hpp)

## SDDMMlib
As you can see to call the SDDMM function we need a function pointer to the actual SDDMM function. This is because we want to benchmark different implementations of the SDDMM algorithm. So you can make different instances of the SDDMMlib class. Then you can just change the function pointers and use the same matrices. The basic (non extensive) interface is as follows:
```cpp
virtual void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const override;
```
To include the function you have to bind it. For example:
```cpp
std::bind(&naive_SDDMM<double>::SDDMM, naive_SDDMM<double>(), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4)
```
This is a bit ugly. TODO: Find a better way to do this.

You can see running examples in the test folder.
