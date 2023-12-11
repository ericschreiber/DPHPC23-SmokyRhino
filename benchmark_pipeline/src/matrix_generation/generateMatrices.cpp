// TODO: maybe also parametrize the type of the matrix (float, double, int, etc.)

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"

// generates a random non-zero number in the range of our matrix and sets it at the given position
void gen_and_set(COOMatrix<float> &matrix, int i, int j)
{
    // generate a random number in the range ]0,1]
    // i.e. the number can't be zero since we are sure that we want to insert a non-zero value)
    float value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    while (value == 0)  // could probably done more efficiently but the chance of getting a 0 (which would make us enter this loop) is very very low
    {
        value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    std::vector<float> oldValues = matrix.getValues();
    oldValues.push_back(value);
    matrix.setValues(oldValues);

    std::vector<int> oldRowIndices = matrix.getRowArray();
    oldRowIndices.push_back(i);
    matrix.setRowArray(oldRowIndices);

    std::vector<int> oldColIndices = matrix.getColIndices();
    oldColIndices.push_back(j);
    matrix.setColIndices(oldColIndices);
}

// generates a random non-zero number in the range of our matrix and sets it at the given position
void gen_and_set(DenseMatrix<float> &matrix, int i, int j)
{
    // generate a random number in the range ]0,1]
    // i.e. the number can't be zero since we are sure that we want to insert a non-zero value)
    float value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    while (value == 0)  // could probably done more efficiently but the chance of getting a 0 (which would make us enter this loop) is very very low
    {
        value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    matrix.setValue(i, j, value);
}

void gen_dense_matrix(DenseMatrix<float> &matrix)
{
    int n = matrix.getNumRows();
    int m = matrix.getNumCols();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)  // TODO: maybe parallelize this loop
        {
            gen_and_set(matrix, i, j);
        }
    }
}

void gen_coo_matrix(COOMatrix<float> &matrix, float sparsity)
{
    int n = matrix.getNumRows();
    int m = matrix.getNumCols();
    // populate array with uniformly distributed random numbers (respecting the degree of sparsity)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            // generate random number in [0,1]
            float decision_num = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (decision_num <= sparsity)
            {
                gen_and_set(matrix, i, j);
            }
        }
    }
}

// since writeToFile expects the dst_file to already exist this script expects this too
int main(int argc, char *argv[])
{
    // check if command line arguments exist
    if (argc != 5)
    {
        std::cout << "Usage: ./generateMatrices <n> <m> <sparsity> <dst_path>" << std::endl;
        return 1;
    }

    // set fresh seed for random number generator (time(0) is too coarse since we invoke this script multiple times in very quick succession)
    uint64_t time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    srand(time);

    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);
    float sparsity = std::stof(argv[3]);
    std::string dst_path = argv[4];

    assert(sparsity >= 0 && sparsity <= 1 && "sparsity must be in [0,1]");

    if (sparsity == 1)
    {
        DenseMatrix<float> matrix(n, m);
        gen_dense_matrix(matrix);
        matrix.writeToFile(dst_path);
        return 0;
    }
    else
    {
        COOMatrix<float> matrix(n, m);
        gen_coo_matrix(matrix, sparsity);
        matrix.writeToFile(dst_path);
        return 0;
    }

    /*
    // test matrix read/write
    COOMatrix<float> matrix2;
    matrix2.readFromFile(dst_path);
    for (int runner = 0; runner < matrix2.getValues().size(); runner++)
    {
        std::cout << matrix2.getValues()[runner] << " " << matrix2.getRowArray()[runner] << " " << matrix2.getColIndices()[runner] << std::endl;
    }
    */
}