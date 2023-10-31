// DenseMatrix.cpp
#include "DenseMatrix.hpp"

#include <cstdlib>
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
    this->numRows = csrMatrix.getNumRows();
    this->numCols = csrMatrix.getNumCols();
    std::vector<std::vector<T>> vals(this->numRows, std::vector<T>(this->numCols, 0));

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

    this->values = vals;
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
// The unnecesary extra storage created for a MxK matrix given M>K is M*K-K^2. So bad for long and thin matrices.
// Improve by inserting and cuting out as you go.
template <typename T>
void DenseMatrix<T>::transpose()
{
    T min = std::min(this->numCols, this->numRows);
    T temp;

    
    if (this->numRows < this->numCols)
    {  // append rows
        for (int i = this->numRows; i < this->numCols; i++)
        {
            this->values.push_back(std::vector<T>());

            for (int j = 0; j < this->numRows; j++)
            {
                this->values[i].push_back(this->values[j][i]);
            }
        }

        // delete col elements in row:
        for (int i = 0; i < this->numRows; i++)
        {
            for (int j = this->numRows; j < this->numCols; j++)
            {
                this->values[i].pop_back();
            }
        }
    }
    else if (this->numCols < this->numRows)
    {
        // append end numbers:
        for (int i = 0; i < this->numCols; i++)
        {
            for (int j = this->numCols; j < this->numRows; j++)
            {
                this->values[i].push_back(this->values[j][i]);
            }
        }
        // remove end rows
        for (int i = this->numCols; i < this->numRows; i++)
        {
            this->values.pop_back();
        }
    }

    // Shift core
    for (int i = 0; i < min; i++)
    {
        for (int j = i + 1; j < min; j++)
        {
            temp = this->values[i][j];
            this->values[i][j] = this->values[j][i];
            this->values[j][i] = temp;
        }
    }

    temp = this->numCols;
    this->numCols = this->numRows;
    this->numRows = temp;
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