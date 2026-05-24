#include <iostream>
#include <random>
#include <string>
#include <functional>

#include "MatrixException.h"

#pragma once
using namespace std;

template <typename T>
class Matrix {
public:
    Matrix(int rSize, int cSize) : rows(rSize), cols(cSize), matSize(rSize * cSize), mat(new T[rSize * cSize]) {}

    Matrix(int rSize, int cSize, T arr[]) : rows(rSize), cols(cSize), matSize(rSize * cSize), mat(new T[rSize * cSize]) {
        for (int i = 0; i < matSize; i++) {
            mat[i] = arr[i];
        }
    }

    // Default constructor for use in Dataset initialization
    Matrix() : rows(-1), cols(-1), matSize(-1), mat(new T[1]) {}

    // Copy constructor
    Matrix(const Matrix& other) {
        rows = other.rows;
        cols = other.cols;
        matSize = other.matSize;

        mat = new T[matSize];
        for (int i = 0; i < matSize; i++) {
            mat[i] = other.mat[i];
        }
    }

    ~Matrix() {
        delete [] mat;
    }

    T& get(int row, int col) {
        checkValidIndex(row, col);
        return mat[row * cols + col];
    }

    Matrix getRow(int row) {
        Matrix mat(1, getColSize());
        for (int c = 0; c < getColSize(); c++) {
            mat.get(0, c) = get(row, c);
        }
        return mat;
    }

    Matrix<T> getCol(int col) {
        Matrix mat(getRowSize(), 1);
        for (int r = 0; r < getRowSize(); r++) {
            mat.get(r, 0) = get(r, col);
        }
        return mat;
    }

    int getRowSize() {
        return rows;
    }

    int getColSize() {
        return cols;
    }

    int getSize() {
        return matSize;
    }

    void randomize() {
        randomize(-10, 10);
    }

    // Assigns random values to all matrix elements
    void randomize(int lowBound, int highBound) {
        mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist(lowBound, highBound);
        for (int i = 0; i < matSize; i++) {
            mat[i] = dist(rng);
        }
    }

    // Copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this == &other) {
            return *this;
        }

        delete [] mat;

        rows = other.rows;
        cols = other.cols;
        matSize = other.matSize;

        mat = new T[matSize];
        for (int i = 0; i < matSize; i++) {
            mat[i] = other.mat[i];
        }

        return *this;
    }

    bool operator==(Matrix mat2) {
        if (getRowSize() != mat2.getRowSize() || getColSize() != mat2.getColSize()) {
            return false;
        }
        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                if (get(r, c) != mat2.get(r, c)) {
                    return false;
                }
            }
        }
        return true;
    }

    Matrix operator+(Matrix mat2) {
        if (getRowSize() != mat2.getRowSize() || getColSize() != mat2.getColSize()) {
            string errMsg1 = "Incorrect dimensions: can't add matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getRowSize(), getColSize());

        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                newMat.get(r, c) = get(r, c) + mat2.get(r, c);
            }
        }

        return newMat;
    }

    Matrix operator-(Matrix mat2) {
        if (getRowSize() != mat2.getRowSize() || getColSize() != mat2.getColSize()) {
            string errMsg1 = "Incorrect dimensions: can't subtract matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getRowSize(), getColSize());

        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                newMat.get(r, c) = get(r, c) - mat2.get(r, c);
            }
        }

        return newMat;
    }

    Matrix operator*(Matrix mat2) {
        if (getRowSize() != mat2.getRowSize() || getColSize() != mat2.getColSize()) {
            string errMsg1 = "Incorrect dimensions: can't perform element-wise multiplication on matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getRowSize(), getColSize());

        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                newMat.get(r, c) = get(r, c) * mat2.get(r, c);
            }
        }

        return newMat;
    }

    Matrix operator/(Matrix mat2) {
        if (getRowSize() != mat2.getRowSize() || getColSize() != mat2.getColSize()) {
            string errMsg1 = "Incorrect dimensions: can't perform element-wise division on matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getRowSize(), getColSize());

        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                newMat.get(r, c) = get(r, c) / mat2.get(r, c);
            }
        }

        return newMat;
    }

    template <typename U>
    void updateAll(function<T(T, U)> f, U val) {
        // f(T, T) takes in the value in the original matrix and a new value and performs some operation

        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                get(r, c) = f(get(r, c), val);
            }
        }
    }

    template <typename U>
    Matrix operator+(U scalar) {
        Matrix newMat = *this;

        function add = [](T x, U y) {return x + y;};
        newMat.updateAll(add, scalar);
        return newMat;
    }

    template <typename U>
    Matrix operator-(U scalar) {
        return operator+(-scalar);
    }

    template <typename U>
    Matrix operator*(U scalar) {
        Matrix newMat = *this;

        function mult = [](T x, U y) {return x * y;};
        newMat.updateAll(mult, scalar);
        return newMat;
    }

    template <typename U>
    Matrix operator/(U scalar) {
        Matrix newMat = *this;

        function div = [](T x, U y) {return x / y;};
        newMat.updateAll(div, scalar);
        return newMat;
    }

    Matrix matMul(Matrix mat2) {
        if (getColSize() != mat2.getRowSize()) {
            string errMsg1 = "Incorrect dimensions: can't multiply matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getRowSize(), mat2.getColSize());

        double dotProduct = 0;
        for (int r1 = 0; r1 < getRowSize(); r1++) {     // Iterate over the rows of mat1
            for (int c2 = 0; c2 < mat2.getColSize(); c2++) {    // Iterate over the cols of mat2
                for (int c1 = 0; c1 < getColSize(); c1 ++) {    // Iterate over the cols of mat1/the rows of mat2
                    dotProduct += get(r1, c1) * mat2.get(c1, c2);
                }
                newMat.get(r1, c2) = dotProduct;
                dotProduct = 0;
            }
        }

        return newMat;
    }

    Matrix transpose() {
        Matrix newMat = Matrix(getColSize(), getRowSize());

        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                newMat.get(c, r) = get(r, c);
            }
        }

        return newMat;
    }



    string toString() {
        string matString = "[";
        for (int r = 0; r < getRowSize(); r++) {
            for (int c = 0; c < getColSize(); c++) {
                matString += to_string(get(r, c));
                if (c < getColSize() - 1) {
                    matString += ", ";
                } else if (r < getRowSize() - 1) {
                    matString += "\n";
                }
            }
        }
        matString += "]";

        return matString;
    }

    void display() {
        cout << toString() << endl;
    }

    string dims() {
        string p1 = "(";
        string c = ", ";
        string p2 = ")";
        return p1 + to_string(rows) + c + to_string(cols) + p2;
    }

private:
    T *mat;
    int rows;
    int cols;
    int matSize;

    void checkValidIndex(int row, int col) {
        if (row >= rows) {
            string errMsg1 = "Row index ";
            string errMsg2 = " is beyond the bounds of the matrix";
            throw MatrixException(errMsg1 + to_string(row) + errMsg2);
        }

        if (col >= cols) {
            string errMsg1 = "Column index ";
            string errMsg2 = " is beyond the bounds of the matrix";
            throw MatrixException(errMsg1 + to_string(col) + errMsg2);
        }
    }

protected:

    void setSize(int rowNum, int colNum) {
        rows = rowNum;
        cols = colNum;
        matSize = rows * cols;

        delete [] mat;
        mat = new T[rows * cols];
    }

};

template <typename T>
string to_string(Matrix<T> mat) {
    return mat.toString();
}

// Extra operator functions to implement commutative operators
template <typename T, typename U>
Matrix<T> operator+(T scalar, Matrix<U> mat) {
    return mat + scalar;
}

template <typename T, typename U>
Matrix<T> operator*(U scalar, Matrix<T> mat) {
    return mat * scalar;
}

// Overloaded mathematical functions for Matrix() objects
template <typename T, typename U>
Matrix<T> pow(Matrix<T> mat, U val) {
    Matrix newMat = mat;

    // function add = [](T x, T y) {return x + y;};
    function customPow = [](T x, U y) {return pow(x, y);};
    newMat.updateAll(customPow, val);
    return newMat;
}

// template <typename T, typename U>
// Matrix<T> pow(Matrix<T> mat, U val) {
//     Matrix newMat = mat;
//
//     for (int r = 0; r < newMat.getRowSize(); r++) {
//         for (int c = 0; c < newMat.getColSize(); c++) {
//             newMat.get(r, c) = pow(newMat.get(r, c), val);
//         }
//     }
//
//     return newMat;
// }

// TODO--implement log, log10, log2, exp, sin, cos
// TODO--change updateAll() to map() that applies the function directly to each value without needing an extra argument?