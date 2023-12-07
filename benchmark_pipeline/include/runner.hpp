#ifndef RUNNER_HPP
#define RUNNER_HPP

#include <dataset_paths.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class runner
{
    public:
        runner(std::string config_file_path, std::string out_path);
        ~runner();
        void run();

    private:
        void write_result();
        void init_result_file();

        bool test_function(const SparseMatrix<T>* const matrixA, const DenseMatrix<T>& matrixB, const DenseMatrix<T>& matrixC, const SDDMMlib<T>* const sddmm_to_run, const std::string sparse_matrix_class, const std::string function_class, const std::string sparse_matrix_path);

        std::string _out_path;
        std::string _results_file_path;
        // A list of tuples to be run. Each tuple contains:
        // - The name of the function class to be run
        // - The name of the sparse matrix type to be used
        // - an instance of dataset_paths
        std::vector<std::tuple<std::string, std::string, dataset_paths>> _functions_to_run;
        // A list of tuples containing the results of the runs. Each tuple contains:
        // - The name of the function class that was run
        // - The name of the sparse matrix type that was used
        // - The name of the dataset that was used
        // - Where the results are stored
        std::vector<std::tuple<std::string, std::string, dataset_paths, std::vector<double>>> _results;
};

#endif  // RUNNER_HPP
