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

    COOMatrix<float> matrix(n, m);

    // populate array with uniformly distributed random numbers (respecting the degree of sparsity)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (sparsity == 0)
            {
                // if sparsity is 0 we want a completely empty matrix (see def of sparsity in Notion).
                // we have to handle this case separately and cant just re-use the else block i.e. generate a decision_num and check if it is <= 0 since decision_num
                // could be exactly 0 (resulting in a matrix with some non-zeros which is not what we want in case of sparsity = 0)
                // we also can't change the condition to < sparsity since then the case where sparsity is 1 could result in a matrix with some zeros
                // (in case decision_num happened to be exactly 1) but by the defintion of sparsity we want a completely full matrix in the case of sparsity = 1.
                continue;
            }
            else
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

    matrix.writeToFile(dst_path);

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