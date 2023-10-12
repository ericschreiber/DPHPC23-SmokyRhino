// DenseMatrix.hpp
#ifndef DENSEMATRIX_HPP
#define DENSEMATRIX_HPP

#include <vector>
#include <string>

template <typename T>
class DenseMatrix {
    public:
        // Constructors
        DenseMatrix(int rows, int cols);  // Constructor for an empty dense matrix
        DenseMatrix(const std::vector<std::vector<T>>& values);  // Copy constructor

        int getNumRows() const;
        int getNumCols() const;
        T at(int row, int col) const;
        void setValue(int row, int col, T value);

        void readFromFile(const std::string& filePath);
        void writeToFile(const std::string& filePath) const;

        T operator[](int row, int col) const;
        
    private:
        std::vector<std::vector<T>> values;
        int numRows;
        int numCols;

};

#endif // DENSEMATRIX_HPP