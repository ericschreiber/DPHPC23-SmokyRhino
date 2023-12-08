// TODO: maybe also parametrize the type of the matrix (float, double, int, etc.)

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"

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

    COOMatrix<float> matrix;
    matrix.setNumRows(n);
    matrix.setNumCols(m);

    // populate array with uniformly distributed random numbers (respecting the degree of sparsity)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            // generate random number between 0 and 1
            float decision_num = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (decision_num < sparsity)
            {
                // generate a random number in the range of out matrix (atm this range is [0,1])
                float value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

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