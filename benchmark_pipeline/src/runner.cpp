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
    // Create a results file with results_timestamp.csv
    _results_file_path = _out_path + "/results_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".csv";
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
    init_result_file();
    // Create the matrices
    CSRMatrix<T> matrixA = CSRMatrix<T>();
    DenseMatrix<T> matrixB = DenseMatrix<T>();
    DenseMatrix<T> matrixC = DenseMatrix<T>();
    CSRMatrix<T> calculatedSolution;

    dataset_paths dataset_before = dataset_paths();
    for (auto& function_to_run : _functions_to_run)
    {
        // Get the function class and dataset
        std::string function_class = std::get<0>(function_to_run);
        dataset_paths dataset = std::get<1>(function_to_run);
        SDDMMlib<T>* class_to_run = get_implemented_class<T>(function_class);
        // Check if the dataset has changed
        if (dataset != dataset_before)
        {
            matrixA.readFromFile(dataset.SparseMatrix_path);
            matrixB.readFromFile(dataset.DenseMatrixA_path);
            matrixC.readFromFile(dataset.DenseMatrixB_path);
            dataset_before = dataset;
        }
        // Time the function
        ExecutionTimer timer = ExecutionTimer();
        class_to_run->set_timer(&timer);

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

        auto durations = timer.get_runs();
        // Append the result to the results list
        _results.push_back(std::make_tuple(function_class, dataset, durations));
        write_result();
    }
}

template <typename T>
void runner<T>::init_result_file()
{
    // Create the results file
    std::ofstream results_file(_results_file_path);
    results_file << "function,path/to/first/dense/matrix,path/to/second/dense/matrix,path/to/first/sparse/matrix,result" << std::endl;
    results_file.close();
}

template <typename T>
void runner<T>::write_result()
{
    // Write the last result to the results file
    // The structure is string, string, string, string, vector<double>
    auto last_result = _results.back();
    std::string durations_string = "";
    for (auto& duration : std::get<2>(last_result))
    {
        durations_string += std::to_string(duration) + ",";
    }
    dataset_paths dataset = std::get<1>(last_result);
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