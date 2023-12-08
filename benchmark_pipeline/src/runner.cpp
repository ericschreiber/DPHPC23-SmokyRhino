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
COOMatrix<T>* convert_to_COO(SparseMatrix<T>* matrix)
{
    // Convert the matrix to CSR if possible else convert to COO
    COOMatrix<T>* coo_matrix = dynamic_cast<COOMatrix<T>*>(matrix);
    if (coo_matrix == nullptr)
    {
        CSRMatrix<T>* csr_matrix = dynamic_cast<CSRMatrix<T>*>(matrix);
        if (csr_matrix == nullptr)
        {
            std::cout << "Error: the matrix is not a CSR or COO matrix" << std::endl;
            return nullptr;
        }
        else
        {
            coo_matrix = new COOMatrix<T>(*csr_matrix);
            delete csr_matrix;
            csr_matrix = nullptr;
        }
    }
    matrix = nullptr;
    return coo_matrix;
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
SparseMatrix<T>* runner<T>::execute_function(
    COOMatrix<T>& matrixA_coo_loader,
    DenseMatrix<T>& matrixB,
    DenseMatrix<T>& matrixC,
    SDDMMlib<T>* sddmm_to_run,
    std::string sparse_matrix_class,
    int num_iterations)
{
    // Create the necessary items
    SparseMatrix<T>* result = get_implemented_SparseMatrix_from_coo<T>(sparse_matrix_class, matrixA_coo_loader.getNumRows(), matrixA_coo_loader.getNumCols());
    SparseMatrix<T>* matrixA = get_implemented_SparseMatrix_from_coo<T>(sparse_matrix_class, matrixA_coo_loader);

    // Print the name of the type of sddmm_to_run using typename
    std::cout << "Starting computation of " << typeid(*sddmm_to_run).name() << std::endl;
    std::cout << "matrixA type: " << typeid(*matrixA).name() << std::endl;
    std::cout << "result type: " << typeid(*result).name() << std::endl;
    std::cout << "matrixA shape: " << matrixA->getNumRows() << "x" << matrixA->getNumCols() << std::endl;
    std::cout << "result shape: " << result->getNumRows() << "x" << result->getNumCols() << std::endl;
    // print the values of matrixA
    std::cout << "matrixA values: ";
    for (auto& value : matrixA->getValues())
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    std::vector<int> rowIndices = matrixA->getRowArray();
    std::cout << "matrixA rowIndices length: " << rowIndices.size() << std::endl;
    std::cout << "matrixA rowIndices: ";
    for (auto& value : rowIndices)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    std::vector<int> colIndices = matrixA->getColIndices();
    std::cout << "matrixA colIndices length: " << colIndices.size() << std::endl;
    std::cout << "matrixA colIndices: ";
    for (auto& value : colIndices)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Running the SDDMM
    matrixA->SDDMM(
        matrixB,
        matrixC,
        *result,
        num_iterations,
        std::bind(
            &SDDMMlib<T>::SDDMM,
            sddmm_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));

    delete matrixA;
    matrixA = nullptr;

    return result;
}

template <typename T>
bool runner<T>::test_function(
    COOMatrix<T>& matrixA_coo_loader,
    DenseMatrix<T>& matrixB,
    DenseMatrix<T>& matrixC,
    std::string sparse_matrix_class,
    std::string function_class)
{
    // Create a CPU timer (we only need this for running it, we don't actually write the measures to the results file)
    ExecutionTimer timer_for_testing = ExecutionTimer();

    // Get the function class
    SDDMMlib<T>* sddmm_to_run = get_implemented_SDDMM<T>(function_class);
    sddmm_to_run->set_timer(&timer_for_testing);
    SDDMMlib<T>* sddmm_standard = new naive_SDDMM_GPU<T>(&timer_for_testing);

    // Run the function
    SparseMatrix<T>* result_standard = execute_function(
        matrixA_coo_loader,
        matrixB,
        matrixC,
        sddmm_standard,
        "CSRMatrix",
        num_iterations_testing);

    SparseMatrix<T>* result_to_test = execute_function(
        matrixA_coo_loader,
        matrixB,
        matrixC,
        sddmm_to_run,
        sparse_matrix_class,
        num_iterations_testing);

    // Print the results
    std::cout << "Result to test: ";
    std::vector<T> result_to_test_values = result_to_test->getValues();
    std::cout << "Result to test length: " << result_to_test_values.size() << std::endl;
    for (auto& value : result_to_test_values)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
    std::cout << "Result standard: ";
    std::vector<T> result_standard_values = result_standard->getValues();
    std::cout << "Result standard length: " << result_standard_values.size() << std::endl;
    for (auto& value : result_standard_values)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    // Checking if the two versions actually returned correctly
    bool correct = check_correctness_and_del_matrices(*result_to_test, *result_standard);

    if (!correct)
    {
        std::cout << "Error: the two versions did not return the same result" << std::endl;
    }

    delete sddmm_to_run;
    sddmm_to_run = nullptr;
    delete sddmm_standard;
    sddmm_standard = nullptr;

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

        // Test the function
        bool test_passed = test_function(matrixA_coo_loader, matrixB, matrixC, sparse_matrix_class, function_class);

        if (test_passed)
        {
            //  Actually do the profiling
            // Time the GPU function
            SDDMMlib<T>* sddmm_to_run = get_implemented_SDDMM<T>(function_class);
            ExecutionTimer timer = ExecutionTimer();
            sddmm_to_run->set_timer(&timer);

            SparseMatrix<T>* result = execute_function(
                matrixA_coo_loader,
                matrixB,
                matrixC,
                sddmm_to_run,
                sparse_matrix_class,
                num_iterations_profiling);

            auto durations = timer.get_runtimes();
            // Append the result to the results list
            _results.push_back(std::make_tuple(function_class, sparse_matrix_class, dataset, durations));
            write_result();

            delete sddmm_to_run;
            sddmm_to_run = nullptr;
            delete result;
            result = nullptr;
        }
        else
        {
            std::cout << "Error: the test did not pass" << std::endl;
            _results.push_back(std::make_tuple("** Function did not pass the test ** " + function_class, sparse_matrix_class, dataset, std::vector<double>()));
            write_result();
        }
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