// DenseMatrix.cpp
#include "DenseMatrix.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "CSRMatrix.hpp"

template <typename T>
DenseMatrix<T>::DenseMatrix()
{
    // Default constructor
    values = nullptr;
    numRows = 0;
    numCols = 0;
}

template <typename T>
DenseMatrix<T>::DenseMatrix(
    int rows,
    int cols) : numRows(rows),
                numCols(cols),
                values(new T[rows * cols])
{
    // Initialize DenseMatrix with zeros
    for (int i = 0; i < rows * cols; ++i)
    {
        values[i] = 0;
    }
}

template <typename T>
DenseMatrix<T>::DenseMatrix(
    const std::vector<std::vector<T>>& values)
{
    this->numRows = values.size();
    this->numCols = values[0].size();
    this->values = new T[this->numRows * this->numCols];

    // copy values
    for (int i = 0; i < this->numRows; i++)
    {
        for (int j = 0; j < this->numCols; j++)
        {
            this->values[i * this->numCols + j] = values[i][j];
        }
    }
}

template <typename T>
DenseMatrix<T>::DenseMatrix(const DenseMatrix<T>& denseMatrix)
{
    this->numRows = denseMatrix.getNumRows();
    this->numCols = denseMatrix.getNumCols();
    this->values = new T[this->numRows * this->numCols];
    // Copy the values
    for (int i = 0; i < this->numRows; ++i)
    {
        for (int j = 0; j < this->numCols; ++j)
        {
            this->values[i * this->numCols + j] = denseMatrix.at(i, j);
        }
    }
}

template <typename T>
DenseMatrix<T>::~DenseMatrix()
{
    // Delete the values
    delete[] values;
}

template <typename T>
DenseMatrix<T>::DenseMatrix(const SparseMatrix<T>& sparseMatrix)
{
    // Check if SparseMatrix is a CSRMatrix
    const CSRMatrix<T>* csrMatrix = dynamic_cast<const CSRMatrix<T>*>(&sparseMatrix);
    if (csrMatrix == nullptr)
    {
        throw std::invalid_argument("Error: DenseMatrix::DenseMatrix(SparseMatrix<T>& sparseMatrix) only accepts CSRMatrix<T> as input");
    }
    else
    {
        convert_csr_dense(*csrMatrix);
    }
}

// constructor to convert CSR matrix to dense matrix
template <typename T>
void DenseMatrix<T>::convert_csr_dense(const CSRMatrix<T>& csrMatrix)
{
    this->numRows = csrMatrix.getNumRows();
    this->numCols = csrMatrix.getNumCols();
    delete[] this->values;
    this->values = new T[this->numRows * this->numCols];
    // initialize values with zeros
    for (int i = 0; i < this->numRows * this->numCols; i++)
    {
        this->values[i] = 0;
    }

    // main loop
    const std::vector<int>& rowIndices = csrMatrix.getRowArray();
    const std::vector<int>& columnIndices = csrMatrix.getColIndices();
    const std::vector<T>& CSR_values = csrMatrix.getValues();

    for (int rowIndex = 0; rowIndex < csrMatrix.getNumRows(); rowIndex++)
    {
        for (int colIndex = rowIndices[rowIndex]; colIndex < rowIndices[rowIndex + 1]; colIndex++)
        {
            this->values[rowIndex * this->numCols + columnIndices[colIndex]] = CSR_values[colIndex];
        }
    }
}

template <typename T>
int DenseMatrix<T>::getNumRows() const
{
    return numRows;
}

template <typename T>
int DenseMatrix<T>::getNumCols() const
{
    return numCols;
}

// added this, don't see why we should not have it
template <typename T>
const T* DenseMatrix<T>::getValues() const
{
    return values;
}

template <typename T>
T DenseMatrix<T>::at(int row, int col) const
{
    // check out of range
    if (row < 0 || row >= numRows || col < 0 || col >= numCols)
    {
        throw std::out_of_range("Error: DenseMatrix::at() out of range");
    }
    else
    {
        return values[row * numCols + col];
    }
}

template <typename T>
void DenseMatrix<T>::setValue(int row, int col, T value)
{
    // check out of range
    if (row < 0 || row >= numRows || col < 0 || col >= numCols)
    {
        throw std::out_of_range("Error: DenseMatrix::setValue() out of range");
    }
    else
    {
        values[row * numCols + col] = value;
    }
}

template <typename T>
void DenseMatrix<T>::setValues(const T* values, int size)
{
    if (size != this->numRows * this->numCols)
    {
        throw std::invalid_argument("Error: DenseMatrix::setValues() size does not match");
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            this->values[i] = values[i];
        }
    }
}

// Should this method ever become too slow/consume to much memory implement this:
// "Use the fact that you are switching positions of only two variables. So you only need a variable to save one, copy the other and save the intermediate to the first position."
template <typename T>
void DenseMatrix<T>::transpose()
{
    T* transposedVals = new T[this->numRows * this->numCols];
    for (int i = 0; i < this->numRows; i++)
    {
        for (int j = 0; j < this->numCols; j++)
        {
            transposedVals[j * this->numRows + i] = this->values[i * this->numCols + j];
        }
    }
    delete[] this->values;
    this->values = transposedVals;
    int temp = this->numRows;
    this->numRows = this->numCols;
    this->numCols = temp;
}

template <typename T>
void DenseMatrix<T>::readFromFile(const std::string& filePath)
{
    std::ifstream file(filePath);
    assert(file.is_open() && "Error: Could not open file for reading");

    std::string datatype;

    // Read numRows, numCols, datatype and they are separated by a comma
    file >> numRows >> numCols >> datatype;
    delete[] this->values;
    this->values = new T[this->numRows * this->numCols];

    // Check the datatype
    if (datatype != typeid(T).name())
    {
        std::cerr << "Error: Datatype in file does not match the datatype of the matrix. Is: " << datatype << " Should be: " << typeid(T).name() << std::endl;
        assert(false);
    }

    // Read the values
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; j++)
        {
            if (!(file >> this->values[i * this->numCols + j]))
            {
                std::cerr << "Error: Could not read value from file even though more should be there." << std::endl;
                return;
            }
        }
    }

    file.close();
}

template <typename T>
void DenseMatrix<T>::writeToFile(const std::string& filePath) const
{
    // The file format is:
    //     numRows numCols datatype
    //     value value value ...
    std::ofstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for writing: " << filePath << std::endl;
        return;
    }

    // Write numRows, numCols, datatype
    file << numRows << " " << numCols << " " << typeid(T).name() << std::endl;

    // Write the values
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; j++)
        {
            file << values[i * this->numCols + j] << " ";
        }
        file << std::endl;
    }

    file.close();
}

// Instanciate the template class with all the types we want to support
template class DenseMatrix<int>;
template class DenseMatrix<float>;
template class DenseMatrix<double>;