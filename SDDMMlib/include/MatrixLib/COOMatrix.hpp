#ifndef COO_MATRIX_HPP
#define COO_MATRIX_HPP

#include <string>
#include <vector>

// if I don't include these here I am unable to use the functions
// in the implementation of this class (it does not make sense to me
// that I have to make these imports here and not in the cpp file
// but I am just happy that I am not getting errors anymore)
#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class COOMatrix : virtual public SparseMatrix<T>
{
    public:
        // constructors
        COOMatrix();                              // Default constructor
        COOMatrix(int rows, int cols);            // Constructor for an empty COO matrix
        COOMatrix(const DenseMatrix<T>& values);  // Constructor from dense matrix
        // SDDMM
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            SparseMatrix<T>& result,
            std::function<void(
                const DenseMatrix<T>& x,
                const DenseMatrix<T>& y,
                const SparseMatrix<T>& z,
                SparseMatrix<T>& result)> SDDMMFunc)
            const override;
        // getters
        virtual int getNumRows() const override;
        virtual int getNumCols() const override;
        virtual int getNumValues() const override;
        virtual const std::vector<T>& getValues() const override;
        virtual const std::vector<int>& getRowArray() const override;
        virtual const std::vector<int>& getColIndices() const override;
        // setters
        virtual T at(int row, int col) const override;  // set on element
        virtual void setValues(const std::vector<T>& values) override;
        virtual void setRowArray(const std::vector<int>& rowPtr) override;
        virtual void setColIndices(const std::vector<int>& colIndices) override;
        // file IO
        virtual void readFromFile(const std::string& filePath) override;
        virtual void writeToFile(const std::string& filePath) const override;
        // other
        virtual ~COOMatrix() {}

    private:
        std::vector<T> values;
        std::vector<int> rowIndices;
        std::vector<int> colIndices;
        int numRows;
        int numCols;
        virtual bool isEqual(const SparseMatrix<T>& other) const override;
};

#endif