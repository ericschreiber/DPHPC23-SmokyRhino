// tests for the implmentation of the COO class

#include <cassert>
#include <iostream>
#include <vector>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"

void mainTest()
{
    // test dense matrix constructor
    std::vector<std::vector<int>> in = {{1, 2, 0}, {0, 5, 6}, {0, 0, 9}};
    COOMatrix<int> coo = COOMatrix<int>(DenseMatrix<int>(in));
    std::vector<int> values = coo.getValues();
    std::vector<int> rowIndices = coo.getRowPtr();
    std::vector<int> colIndices = coo.getColIndices();
    assert(values.size() == 5);
    assert(rowIndices.size() == 5);
    assert(colIndices.size() == 5);
    assert(values[0] == 1);
    assert(values[1] == 2);
    assert(values[2] == 5);
    assert(values[3] == 6);
    assert(values[4] == 9);
    assert(rowIndices[0] == 0);
    assert(rowIndices[1] == 0);
    assert(rowIndices[2] == 1);
    assert(rowIndices[3] == 1);
    assert(rowIndices[4] == 2);
    assert(colIndices[0] == 0);
    assert(colIndices[1] == 1);
    assert(colIndices[2] == 1);
    assert(colIndices[3] == 2);
    assert(colIndices[4] == 2);

    // test at()
    assert(coo.at(0, 0) == 1);
    assert(coo.at(0, 1) == 2);
    assert(coo.at(1, 1) == 5);
    assert(coo.at(1, 2) == 6);
    assert(coo.at(2, 2) == 9);
    assert(coo.at(0, 2) == 0);

    // test the equality operator
    COOMatrix<int> coo2 = COOMatrix<int>(DenseMatrix<int>(in));
    assert(coo == coo2);
    std::vector<std::vector<int>> in2 = {{1, 2, 0}, {0, 5, 6}, {0, 0, 10}};
    COOMatrix<int> coo3 = COOMatrix<int>(DenseMatrix<int>(in2));
    assert((coo == coo3) == false);
}

int main()
{
    mainTest();
    return 0;
}