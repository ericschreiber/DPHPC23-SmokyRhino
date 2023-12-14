#ifndef RUNNER_HPP
#define RUNNER_HPP

#include <dataset_paths.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "COOMatrix.hpp"
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

        bool test_function(
            COOMatrix<T>& matrixA_coo_loader,
            DenseMatrix<T>& matrixB,
            DenseMatrix<T>& matrixC,
            std::string sparse_matrix_class,
            std::string function_class);

        SparseMatrix<T>* execute_function(
            COOMatrix<T>& matrixA_coo_loader,
            DenseMatrix<T>& matrixB,
            DenseMatrix<T>& matrixC,
            SDDMMlib<T>* sddmm_to_run,
            std::string sparse_matrix_class,
            int num_iterations);

        bool do_test = false;

        const int num_iterations_testing = 1;
        const int num_iterations_profiling = 1;
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
