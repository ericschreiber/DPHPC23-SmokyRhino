#include "runner.hpp"

#include <chrono>

#include "COOMatrix.hpp"
#include "CSRMatrix.hpp"
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
CSRMatrix<T>* convert_to_CSR(SparseMatrix<T>* matrix)
{
    // Convert the matrix to CSR if possible else convert to COO
    CSRMatrix<T>* csr_matrix = dynamic_cast<CSRMatrix<T>*>(matrix);
    if (csr_matrix == nullptr)
    {
        COOMatrix<T>* coo_matrix = dynamic_cast<COOMatrix<T>*>(matrix);
        if (coo_matrix == nullptr)
        {
            std::cout << "Error: the matrix is not a CSR or COO matrix" << std::endl;
            return nullptr;
        }
        else
        {
            csr_matrix = new CSRMatrix<T>(*coo_matrix);
            delete coo_matrix;
            coo_matrix = nullptr;
        }
    }
    matrix = nullptr;
    return csr_matrix;
}
template <typename T>
bool check_correctness_and_del_matrices(SparseMatrix<T>& trueSolution, SparseMatrix<T>& calculatedSolution)
{
    // Convert the matrices to CSR if possible else convert to COO
    CSRMatrix<T>* trueSolution_csr = convert_to_CSR(&trueSolution);
    CSRMatrix<T>* calculatedSolution_csr = convert_to_CSR(&calculatedSolution);

    // Check if the two matrices are equal
    bool correct = *trueSolution_csr == *calculatedSolution_csr;

    delete trueSolution_csr;
    trueSolution_csr = nullptr;
    delete calculatedSolution_csr;
    calculatedSolution_csr = nullptr;

    return correct;
}

template <typename T>
bool runner<T>::test_function(const SparseMatrix<T>* const matrixA, const DenseMatrix<T>& matrixB, const DenseMatrix<T>& matrixC, const SDDMMlib<T>* const sddmm_to_run, const std::string sparse_matrix_class, const std::string function_class, const std::string sparse_matrix_path)
{
    // Create a CPU timer (we only need this for running it, we don't actually write the measures to the results file)
    ExecutionTimer timer_for_testing = ExecutionTimer();
    const int num_iterations_testing = 1;

    // Test the version before running the profiling
    SparseMatrix<T>* result_standard = get_implemented_SparseMatrix<T>("CSRMatrix", matrixA->getNumRows(), matrixA->getNumCols());
    SparseMatrix<T>* result_to_test = get_implemented_SparseMatrix<T>(sparse_matrix_class, matrixA->getNumRows(), matrixA->getNumCols());

    SDDMMlib<T>* class_standard = new semi_naive_CSR_SDDMM_GPU<T>(&timer_for_testing);
    // As matrixA test has always to be CSR, we need to convert it to CSR if needed
    COOMatrix<T> CooMatrix = COOMatrix<T>();
    CooMatrix.readFromFile(sparse_matrix_path);
    CSRMatrix<T> csrMatrix = CSRMatrix<T>(CooMatrix);

    // Running the CPU version
    csrMatrix.SDDMM(
        matrixB,
        matrixC,
        *result_standard,
        num_iterations_testing,
        std::bind(
            &SDDMMlib<T>::SDDMM,
            class_standard,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

    delete class_standard;
    class_standard = nullptr;

    // Running the GPU version
    matrixA->SDDMM(
        matrixB,
        matrixC,
        *result_to_test,
        num_iterations_testing,
        std::bind(
            &SDDMMlib<T>::SDDMM,
            sddmm_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

    // Checking if the two versions actually returned correctly
    bool correct = check_correctness_and_del_matrices(*result_to_test, *result_standard);

    if (!correct)
    {
        std::cout << "Error: the two versions did not return the same result" << std::endl;
    }

    return correct;
}

template <typename T>
void runner<T>::run()
{
    const int num_iterations_profiling = 3;  // Currently always +1 execution is timed in the output becuase of testing.
    init_result_file();
    // Create the matrices
    COOMatrix<T> matrixA_coo_loader = COOMatrix<T>();
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
            matrixA_coo_loader.readFromFile(dataset.SparseMatrix_path);
            matrixB.readFromFile(dataset.DenseMatrixA_path);
            matrixC.readFromFile(dataset.DenseMatrixB_path);
            dataset_before = dataset;
        }

        SDDMMlib<T>* sddmm_to_run = get_implemented_SDDMM<T>(function_class);
        SparseMatrix<T>* matrixA = get_implemented_SparseMatrix<T>(sparse_matrix_class, matrixA_coo_loader);
        SparseMatrix<T>* calculatedSolution = get_implemented_SparseMatrix<T>(sparse_matrix_class, matrixA_coo_loader.getNumRows(), matrixC.getNumCols());

        // Time the GPU function
        ExecutionTimer timer = ExecutionTimer();
        sddmm_to_run->set_timer(&timer);

        // Test the function
        bool test_passed = test_function(matrixA, matrixB, matrixC, sddmm_to_run, sparse_matrix_class, function_class, dataset.SparseMatrix_path);

        if (test_passed)
        {
            // Actually do the profiling
            // Run the function
            matrixA->SDDMM(
                matrixB,
                matrixC,
                *calculatedSolution,
                num_iterations_profiling,
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
        }

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