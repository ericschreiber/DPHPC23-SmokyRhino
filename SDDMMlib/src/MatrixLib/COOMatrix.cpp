#include "COOMatrix.hpp"

//////////////// CONSTRUCTORS ////////////////

// this constructor is used to create an empty COO matrix
template <typename T>
COOMatrix<T>::COOMatrix(int rows, int cols)
{
    this->values = std::vector<T>();
    this->rowIndices = std::vector<int>();
    this->colIndices = std::vector<int>();
    // this (in my opinion) is significantly more readable than the
    // syntax where the fields are initialized in the signature
    this->numRows = rows;
    this->numCols = cols;
}

// this constructor is used to convert a dense matrix to COO format
template <typename T>
COOMatrix<T>::COOMatrix(const DenseMatrix<T>& denseMatrix)
{
    this->numRows = denseMatrix.getNumRows();
    this->numCols = denseMatrix.getNumCols();
    // initialize as empty vectors
    this->values = std::vector<T>();
    this->rowIndices = std::vector<int>();
    this->colIndices = std::vector<int>();

    T* valuesPointer = denseMatrix.getValues();
    int numElems = denseMatrix->getNumRows() * denseMatrix->getNumCols();
    for (int i = 0; i < numElems; i++)
    {
        T value = *valuesPointer;
        if (value != 0)
        {
            this->values.push_back(value);
            // generate row and column indices from linear index
            this->rowIndices.push_back(i / this->numCols);  // "/" is integer division
            this->colIndices.push_back(i % this->numCols);
        }
        valuesPointer++;
    }
}

//////////////// SDDMM ////////////////

template <typename T>
void COOMatrix<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    SparseMatrix<T>& result,
    std::function<void(
        const DenseMatrix<T>& x,
        const DenseMatrix<T>& y,
        const SparseMatrix<T>& z,
        SparseMatrix<T>& result)> SDDMMFunc)
    const
{
    // I guess we do the same thing here that we do in CSRMatrix.cpp (?)
    SDDMMFunc(x, y, *this, result);
}

//////////////// GETTERS ////////////////

// returns an int by value
template <typename T>
int COOMatrix<T>::getNumRows() const
{
    return this->numRows;
}

// returns an int by value
template <typename T>
int COOMatrix<T>::getNumCols() const
{
    return this->numCols;
}

// returns an int by value
template <typename T>
int COOMatrix<T>::getNumValues() const
{
    return this->values.size();
}

// returns a reference to the vetor
template <typename T>
const std::vector<T>& COOMatrix<T>::getValues() const
{
    return this->values;
}

// returns a reference to the vector
template <typename T>
const std::vector<int>& COOMatrix<T>::getRowIndices() const
{
    return this->rowIndices;
}

// returns a reference to the vector
template <typename T>
const std::vector<int>& COOMatrix<T>::getColIndices() const
{
    return this->colIndices;
}

//////////////// SETTERS ////////////////

template <typename T>
T COOMatrix<T>::at(int row, int col) const
{
    // check that row and col are not out of bounds
    if (row < 0 || row >= this->numRows)
    {
        throw std::invalid_argument("error: row index out of bounds");
    }
    if (col < 0 || col >= this->numCols)
    {
        throw std::invalid_argument("error: column index out of bounds");
    }

    int runner = 0;
    while (runner < this->values.size())
    {
        int rowIndex = rowIndices[runner];
        int colIndex = colIndices[runner];
        if (rowIndex == row && colIndex == col)
        {
            return values[runner];
        }
    }
    // if we get here the value is 0 (since we are a sparse matrix
    // and we have made sure that row and col are not out of bounds)
    return 0;
}

template <typename T>
void COOMatrix<T>::setValues(const std::vector<T>& values)
{
    this->values = values;
}

template <typename T>
void COOMatrix<T>::setRowIndices(const std::vector<int>& rowIndices)
{
    this->rowIndices = rowIndices;
}

template <typename T>
void COOMatrix<T>::setColIndices(const std::vector<int>& colIndices)
{
    this->colIndices = colIndices;
}

//////////////// FILE IO ////////////////

template <typename T>
void COOMatrix<T>::readFromFile(const std::string& filePath)
{
    // TODO: do we even need this since the matrix generation script is
    // generating CSR matrices (and the matrix generation script is
    // the only place where we want to save matrices to a file
    // (if I am not mistaken))?
}

template <typename T>
void COOMatrix<T>::writeToFile(const std::string& filePath) const
{
    // TODO: same comment as in readFromFile
}

//////////////// OTHER ////////////////

// the ~ thing apparently implements itself...

template <typename T>
bool COOMatrix<T>::operator==(const SparseMatrix<T>& other) const
{
    // compare values arrays
    //
    // since I am using refs the following operations should be in place
    std::vector<T>& valsReference1 = this->getValues();
    std::vector<T>& valsReference2 = other.getValues();
    std::sort(valsReference1.begin(), valsReference1.end());
    std::sort(valsReference2.begin(), valsReference2.end());
    // now both this->values and other.values are sorted since
    // if we call sort on a reference it will modify the original object
    if (this->values != other.values)
    {
        return false;
    }
    // compare row indices arrays
    std::vector<int>& rowIndicesReference1 = this->getRowIndices();
    std::vector<int>& rowIndicesReference2 = other.getRowIndices();
    std::sort(rowIndicesReference1.begin(), rowIndicesReference1.end());
    std::sort(rowIndicesReference2.begin(), rowIndicesReference2.end());
    if (this->rowIndices != other.rowIndices)
    {
        return false;
    }
    // compare col indices arrays
    std::vector<int>& colIndicesReference1 = this->getColIndices();
    std::vector<int>& colIndicesReference2 = other.getColIndices();
    std::sort(colIndicesReference1.begin(), colIndicesReference1.end());
    std::sort(colIndicesReference2.begin(), colIndicesReference2.end());
    if (this->colIndices != other.colIndices)
    {
        return false;
    }
}

// TODO: have the discussion about the SparseMatrix interface
// (and maybe rm the following methods afterwards)

template <typename T>
const std::vector<int>& COOMatrix<T>::getRowPtr() const
{
}

template <typename T>
void COOMatrix<T>::setRowPtr(const std::vector<int>& rowPtr)
{
}
