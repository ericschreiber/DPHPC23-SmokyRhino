// TODO: maybe also parametrize the type of the matrix (float, double, int, etc.)

#include <chrono>
#include <cstdint>
#include <iostream>

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

    // set fresh seed for random number generator
    uint64_t time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    srand(time);

    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);
    float sparsity = std::stof(argv[3]);
    std::string dst_path = argv[4];

    // populate array with uniformly distributed random numbers (respecting the degree of sparsity)
    std::vector<std::vector<float> > vals(n, std::vector<float>(m, 0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (sparsity == 0)  // dense matrix
            {
                float rand_num = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                vals[i][j] = rand_num;
            }
            else
            {  // sparse matrix
                float rand_num_sparsity = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                if (rand_num_sparsity > sparsity)
                {
                    float rand_num = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                    vals[i][j] = rand_num;
                }
            }
            // new line
        }
    }

    // in the meeting we have said that we always want to save the matrix as a dense matrix
    DenseMatrix<float> dense_matrix(vals);
    dense_matrix.writeToFile(dst_path);
    return 0;
}