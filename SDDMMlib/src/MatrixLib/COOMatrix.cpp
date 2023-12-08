#include "COOMatrix.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>

//////////////// CONSTRUCTORS ////////////////
template <typename T>
COOMatrix<T>::COOMatrix()
{
    this->values = std::vector<T>();
    this->rowIndices = std::vector<int>();
    this->colIndices = std::vector<int>();
    this->numRows = 0;
    this->numCols = 0;
}

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

    const T* valuesPointer = denseMatrix.getValues();
    int numElems = denseMatrix.getNumRows() * denseMatrix.getNumCols();
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
const std::vector<int>& COOMatrix<T>::getRowArray() const
{
    return this->rowIndices;
}

// returns a reference to the vector
template <typename T>
const std::vector<int>& COOMatrix<T>::getColIndices() const
{
    return this->colIndices;
}

// I think this is a getter and not a setter
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
        runner++;
    }
    // if we get here the value is 0 (since we are a sparse matrix
    // and we have made sure that row and col are not out of bounds)
    return 0;
}

//////////////// SETTERS ////////////////

template <typename T>
void COOMatrix<T>::setNumRows(int numRows)
{
    this->numRows = numRows;
}

template <typename T>
void COOMatrix<T>::setNumCols(int numCols)
{
    this->numCols = numCols;
}

template <typename T>
void COOMatrix<T>::setValues(const std::vector<T>& values)
{
    this->values = values;
}

template <typename T>
void COOMatrix<T>::setRowArray(const std::vector<int>& rowIndices)
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
    // format is:
    // <rowIndex> <colIndex> <value> \n
    // <rowIndex> <colIndex> <value> \n
    // ...
    std::ifstream file(filePath);
    assert(file.is_open() && "Error: Could not open file for reading");

    int rowIndex, colIndex;
    T value;
    while (file >> rowIndex >> colIndex >> value)
    {
        this->rowIndices.push_back(rowIndex);
        this->colIndices.push_back(colIndex);
        this->values.push_back(value);
    }

    file.close();
}

template <typename T>
void COOMatrix<T>::writeToFile(const std::string& filePath) const
{
    // format is:
    // <rowIndex> <colIndex> <value> \n
    // <rowIndex> <colIndex> <value> \n
    // ...

    std::ofstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filePath << std::endl;
    }

    // Write the actual matrix content
    for (int runner = 0; runner < values.size(); runner++)
    {
        file << rowIndices[runner] << " " << colIndices[runner] << " " << values[runner] << std::endl;
    }

    file.close();
}

//////////////// OTHER ////////////////

// the ~ thing apparently implements itself...

template <typename T>
bool COOMatrix<T>::isEqual(const SparseMatrix<T>& other) const
{
    // compare values arrays
    //
    // if I sort the original vectors c++ will complain, hence I have to make copies

    std::vector<T> valsCopy1 = std::vector<T>();  // create two empty vectors
    std::vector<T> valsCopy2 = std::vector<T>();
    std::copy(this->getValues().begin(), this->getValues().end(), std::back_inserter(valsCopy1));  // copy the original vectors into the empty vectors
    std::copy(other.getValues().begin(), other.getValues().end(), std::back_inserter(valsCopy2));
    std::vector<T>& valsReference1 = valsCopy1;  // get references to the copies
    std::vector<T>& valsReference2 = valsCopy2;
    std::sort(valsReference1.begin(), valsReference1.end());
    std::sort(valsReference2.begin(), valsReference2.end());
    if (valsReference1.size() != valsReference2.size())  // std::equal requires us to ensure that the sizes are equal
    {
        return false;
    }
    else
    {
        // comparing references to vectors is not what we want here so we need to use std::equal
        if (!std::equal(valsReference1.begin(), valsReference1.end(), valsReference2.begin()))
        {
            return false;
        }
    }

    // compare row indices arrays
    std::vector<int> rowIndicesCopy1 = std::vector<int>();
    std::vector<int> rowIndicesCopy2 = std::vector<int>();
    std::copy(this->getRowArray().begin(), this->getRowArray().end(), std::back_inserter(rowIndicesCopy1));
    std::copy(other.getRowArray().begin(), other.getRowArray().end(), std::back_inserter(rowIndicesCopy2));
    std::vector<int>& rowIndicesReference1 = rowIndicesCopy1;
    std::vector<int>& rowIndicesReference2 = rowIndicesCopy2;
    std::sort(rowIndicesReference1.begin(), rowIndicesReference1.end());
    std::sort(rowIndicesReference2.begin(), rowIndicesReference2.end());
    if (rowIndicesReference1.size() != rowIndicesReference2.size())
    {
        return false;
    }
    else
    {
        if (!std::equal(rowIndicesReference1.begin(), rowIndicesReference1.end(), rowIndicesReference2.begin()))
        {
            return false;
        }
    }

    // compare col indices arrays
    std::vector<int> colIndicesCopy1 = std::vector<int>();
    std::vector<int> colIndicesCopy2 = std::vector<int>();
    std::copy(this->getColIndices().begin(), this->getColIndices().end(), std::back_inserter(colIndicesCopy1));
    std::copy(other.getColIndices().begin(), other.getColIndices().end(), std::back_inserter(colIndicesCopy2));
    std::vector<int>& colIndicesReference1 = colIndicesCopy1;
    std::vector<int>& colIndicesReference2 = colIndicesCopy2;
    std::sort(colIndicesReference1.begin(), colIndicesReference1.end());
    std::sort(colIndicesReference2.begin(), colIndicesReference2.end());
    if (colIndicesReference1.size() != colIndicesReference2.size())
    {
        return false;
    }
    else
    {
        if (!std::equal(colIndicesReference1.begin(), colIndicesReference1.end(), colIndicesReference2.begin()))
        {
            return false;
        }
    }

    return true;
}

// we need these declarations for every instantiation of the SparseMatrix interface
// (otherwise the linker will complain that it can't find the implementations)
template class COOMatrix<float>;
template class COOMatrix<double>;
template class COOMatrix<int>;