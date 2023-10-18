#include "runner.hpp"

#include <chrono>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "config_helper.hpp"
#include "implemented_classes.hpp"

template <typename T>
runner<T>::runner(std::string config_file_path, std::string out_path)
{
    read_config_file(config_file_path, _functions_to_run);
    // Create the output directory if it does not exist
    std::string command = "mkdir -p " + out_path;
    system(command.c_str());
    _out_path = out_path;
    _results_file_path = _out_path + "/results.csv";
    std::ofstream results_file(_results_file_path);
    results_file << "function,dataset,result" << std::endl;
    results_file.close();
}

template <typename T>
runner<T>::~runner()
{
}

template <typename T>
void runner<T>::run()
{
    _results.push_back(std::make_tuple("function", "dataset", "result"));
    // Create the matrices
    CSRMatrix<T> matrixA = CSRMatrix<T>();
    DenseMatrix<T> matrixB = DenseMatrix<T>();
    DenseMatrix<T> matrixC = DenseMatrix<T>();
    CSRMatrix<T> calculatedSolution;

    std::string dataset_before = "";
    for (auto& function_to_run : _functions_to_run)
    {
        // Get the function class and dataset
        std::string function_class = std::get<0>(function_to_run);
        std::string dataset = std::get<1>(function_to_run);
        SDDMMlib<T>* class_to_run = get_implemented_class<T>(function_class);
        // Check if the dataset has changed
        if (dataset != dataset_before)
        {
            // Read the dataset
            // ************************************************************
            // TODO: Implement this
            // ************************************************************
            // For the moment we just initialize the matrices here
            std::cout << "!!!!!!PLEASE IMPLEMENT MATRIX LOADING!!!!!!" << std::endl;
            matrixA.readFromFile("/Users/ericschreiber/dev/ETH/HPC_ETH/project/benchmark_pipeline/benchmark_pipeline/tests/csrmatrix_test.txt");
            matrixB.readFromFile("/Users/ericschreiber/dev/ETH/HPC_ETH/project/benchmark_pipeline/benchmark_pipeline/tests/densematrix_test.txt");
            matrixC.readFromFile("/Users/ericschreiber/dev/ETH/HPC_ETH/project/benchmark_pipeline/benchmark_pipeline/tests/densematrix_test.txt");

            dataset_before = dataset;
        }
        // Time the function
        auto start = std::chrono::high_resolution_clock::now();

        // Prepare the calculated solution
        calculatedSolution = CSRMatrix<T>(matrixA.getNumRows(), matrixA.getNumRows());

        // Run the function
        matrixA.SDDMM(
            matrixB,
            matrixC,
            calculatedSolution,
            std::bind(
                &SDDMMlib<T>::SDDMM,
                class_to_run,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4));

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // Append the result to the results list
        _results.push_back(std::make_tuple(function_class, dataset, std::to_string(duration.count())));
        write_result();
    }
}

template <typename T>
void runner<T>::write_result()
{
    // Write the last result to the results file
    std::ofstream results_file(_results_file_path, std::ios::app);
    results_file << std::get<0>(_results.back()) << ","
                 << std::get<1>(_results.back()) << ","
                 << std::get<2>(_results.back()) << std::endl;
}

template class runner<float>;
template class runner<double>;
template class runner<int>;