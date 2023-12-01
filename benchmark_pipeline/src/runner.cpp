#include "runner.hpp"

#include <chrono>

#include "COOMatrix.hpp"
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
    // Create a results file with results_timestamp.csv
    _results_file_path = _out_path + "/results_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".csv";
    std::ofstream results_file(_results_file_path);
    results_file << "function,format,dataset,result" << std::endl;
    results_file.close();
}

template <typename T>
runner<T>::~runner()
{
}

template <typename T>
void runner<T>::run()
{
    const int num_iterations = 10;
    init_result_file();
    // Create the matrices
    DenseMatrix<T> matrixA_dense_loader = DenseMatrix<T>();
    DenseMatrix<T> matrixB = DenseMatrix<T>();
    DenseMatrix<T> matrixC = DenseMatrix<T>();
    // CSRMatrix<T> calculatedSolution;

    dataset_paths dataset_before = dataset_paths();
    for (auto& function_to_run : _functions_to_run)
    {
        // Get the function class and dataset
        std::string function_class = std::get<0>(function_to_run);
        std::string sparse_matrix_class = std::get<1>(function_to_run);
        dataset_paths dataset = std::get<2>(function_to_run);

        // Check if the dataset has changed
        if (dataset != dataset_before)
        {
            matrixA_dense_loader.readFromFile(dataset.SparseMatrix_path);
            matrixB.readFromFile(dataset.DenseMatrixA_path);
            matrixC.readFromFile(dataset.DenseMatrixB_path);
            dataset_before = dataset;
        }

        SDDMMlib<T>* sddmm_to_run = get_implemented_SDDMM<T>(function_class);
        SparseMatrix<T>* matrixA = get_implemented_SparseMatrix<T>(sparse_matrix_class, matrixA_dense_loader);
        SparseMatrix<T>* calculatedSolution = get_implemented_SparseMatrix<T>(sparse_matrix_class, matrixA_dense_loader.getNumRows(), matrixC.getNumCols());

        // Time the function
        ExecutionTimer timer = ExecutionTimer();
        sddmm_to_run->set_timer(&timer);

        // Run the function
        matrixA->SDDMM(
            matrixB,
            matrixC,
            *calculatedSolution,
            num_iterations,
            std::bind(
                &SDDMMlib<T>::SDDMM,
                sddmm_to_run,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5));

        auto durations = timer.get_runtimes();
        // Append the result to the results list
        _results.push_back(std::make_tuple(function_class, sparse_matrix_class, dataset, durations));
        write_result();

        delete sddmm_to_run;
        sddmm_to_run = nullptr;
        delete matrixA;
        matrixA = nullptr;
        delete calculatedSolution;
        calculatedSolution = nullptr;
    }
}

template <typename T>
void runner<T>::init_result_file()
{
    // Create the results file
    std::ofstream results_file(_results_file_path);
    results_file << "function, SparseFormat, path/to/first/dense/matrix, path/to/second/dense/matrix, path/to/first/sparse/matrix, result" << std::endl;
    results_file.close();
}

template <typename T>
void runner<T>::write_result()
{
    // Write the last result to the results file
    // The structure is string, string, string, string, vector<double>
    auto last_result = _results.back();
    std::string durations_string = "";
    for (auto& duration : std::get<3>(last_result))
    {
        durations_string += std::to_string(duration) + ",";
    }
    dataset_paths dataset = std::get<2>(last_result);
    std::string dataset_string = dataset.DenseMatrixA_path + "," +
                                 dataset.DenseMatrixB_path + "," +
                                 dataset.SparseMatrix_path;
    std::ofstream results_file(_results_file_path, std::ios::app);
    results_file << std::get<0>(last_result) << ","
                 << dataset_string << ","
                 << durations_string << std::endl;
}

template class runner<float>;
template class runner<double>;
template class runner<int>;