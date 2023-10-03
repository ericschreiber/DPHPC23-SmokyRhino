# Sparse Matrix
This library implements the sparse matrix data structure. It is a matrix that is composed of mostly zeros. This library is used in the benchmark pipeline and others. You may overwrite the multiply function to implement your own sparse matrix multiplication algorithm.
For each storage type there is a different implementation of the sparse matrix. The implemented storage types are:
- CSR

## Storage Types
### CSR
The CSR type is stored as follows each on a new line in the file.
-     numRows numCols
-     numNonZeros
-     datatype
-     value colIndex
-     ...
-     row pointers

### Dense Matrix
The dense matrix is stored as follows each on a new line in the file.
-     numRows numCols DataType
-     value value value ...

## Usage    
