#ifndef DATASET_PATHS_HPP
#define DATASET_PATHS_HPP

#include <string>

struct dataset_paths
{
        dataset_paths() : DenseMatrixA_path(""), DenseMatrixB_path(""), SparseMatrix_path("") {}
        dataset_paths(std::string DenseMatrixA_path, std::string DenseMatrixB_path, std::string SparseMatrix_path)
            : DenseMatrixA_path(DenseMatrixA_path), DenseMatrixB_path(DenseMatrixB_path), SparseMatrix_path(SparseMatrix_path) {}

        bool operator==(const dataset_paths& other) const
        {
            return DenseMatrixA_path == other.DenseMatrixA_path &&
                   DenseMatrixB_path == other.DenseMatrixB_path &&
                   SparseMatrix_path == other.SparseMatrix_path;
        }
        bool operator!=(const dataset_paths& other) const
        {
            return !(*this == other);
        }
        std::string DenseMatrixA_path;
        std::string DenseMatrixB_path;
        std::string SparseMatrix_path;
};

#endif  // DATASET_PATHS_HPP