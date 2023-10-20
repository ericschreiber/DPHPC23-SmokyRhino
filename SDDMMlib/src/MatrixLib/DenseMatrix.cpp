// DenseMatrix.cpp
#include "DenseMatrix.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "CSRMatrix.hpp"

template <typename T>
DenseMatrix<T>::DenseMatrix(int rows, int cols) : numRows(rows),
                                                  numCols(cols),
                                                  values(rows, std::vector<T>(cols, T()))
{
}

template <typename T>
DenseMatrix<T>::DenseMatrix(const std::vector<std::vector<T>>& values) : numRows(values.size()),
                                                                         numCols(values[0].size()),
                                                                         values(values)
{
}

// constructor to convert CSR matrix to dense matrix
template <typename T>
DenseMatrix<T>::DenseMatrix(CSRMatrix<T>& csrMatrix)
{
    int num_rows = csrMatrix.getNumRows();
    int num_cols = csrMatrix.getNumCols();
    std::vector<std::vector<T>> vals(num_rows, std::vector<T>(num_cols, 0));

    // main loop
    std::vector<int> rowIndices = csrMatrix.getRowPtr();
    std::vector<int> columnIndices = csrMatrix.getColIndices();
    std::vector<T> values = csrMatrix.getValues();
    for (int rowIndicesArrayRunner = 0; rowIndicesArrayRunner < rowIndices.size(); rowIndicesArrayRunner++)
    {
        int num_elems_in_row = rowIndices[rowIndicesArrayRunner + 1] - rowIndices[rowIndicesArrayRunner];
        for (int i = 0; i < num_elems_in_row; i++)
        {
            int index = rowIndices[rowIndicesArrayRunner] + i;  // this index indexes columnIndices and values
            int column_index = columnIndices[index];
            int value = values[index];
            vals[rowIndicesArrayRunner][column_index] = value;
        }
        rowIndicesArrayRunner++;
    }

    // write class fields
    this->values = vals;
    this->numRows = num_rows;
    this->numCols = num_cols;
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
std::vector<std::vector<T>> DenseMatrix<T>::getValues()
{
    return this->values;
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
        return values[row][col];
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
        values[row][col] = value;
    }
}

// Should this method ever become too slow/consume to much memory implement this:
// "Use the fact that you are switching positions of only two variables. So you only need a variable to save one, copy the other and save the intermediate to the first position."
template <typename T>
void DenseMatrix<T>::transpose()
{
    std::vector<std::vector<T>> vals = this->values;
    std::vector<std::vector<T>> transposedVals(this->numCols, std::vector<T>(this->numRows, 0));
    for (int i = 0; i < this->numRows; i++)
    {
        for (int j = 0; j < this->numCols; j++)
        {
            transposedVals[j][i] = vals[i][j];
        }
    }
    this->values = transposedVals;
    int temp = this->numRows;
    this->numRows = this->numCols;
    this->numCols = temp;
}

template <typename T>
void DenseMatrix<T>::readFromFile(const std::string& filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file for reading: " << filePath << std::endl;
        return;
    }

    std::string datatype;

    // Read numRows, numCols, datatype
    file >> numRows >> numCols >> datatype;
    values.resize(numRows, std::vector<T>(numCols, T()));

    // Check the datatype
    if (datatype != typeid(T).name())
    {
        std::cerr << "Error: Datatype in file does not match the datatype of the matrix" << std::endl;
        return;
    }

    // Read the values
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; j++)
        {
            if (!(file >> values[i][j]))
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
            file << values[i][j] << " ";
        }
        file << std::endl;
    }

    file.close();
}

// Instanciate the template class with all the types we want to support
template class DenseMatrix<int>;
template class DenseMatrix<float>;
template class DenseMatrix<double>;