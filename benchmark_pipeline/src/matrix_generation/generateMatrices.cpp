// TODO: maybe also parametrize the type of the matrix (float, double, int, etc.)

// Note: to compile this into a "standalone" executable and not into as part of a library I threw
// this file into tests/MatrixLib and then added the following line to the CMakeLists.txt file:
// under "Link the excecutable" I added: "add_executable(generateMatrices generateMatrices.cpp)"
// under "Link the library" I added: "target_link_libraries(generateMatrices MatrixLib)"

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

    int n = std::stoi(argv[1]);
    int m = std::stoi(argv[2]);
    // maybe this arg can be made optional, for now just set to 0 in case of dense matrix
    float sparsity = std::stof(argv[3]);
    std::string dst_path = argv[4];

    // populate array with uniofrmly distributed random numbers (respecting the degree of sparsity)
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
        }
    }

    // in the meeting we have said that we always want to save the matrix as a dense matrix
    DenseMatrix<float> dense_matrix(vals);
    dense_matrix.writeToFile(dst_path);
    return 0;
}