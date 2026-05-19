// TODO--implement:
// inverse?

#include <iostream>
#include <random>
#include <string>
#include <functional>

// #include "Dual.h"
#include "MatrixException.h"
#pragma once
using namespace std;

template <typename T>
class Matrix {
public:
    Matrix(int rSize, int cSize) : rng(random_device{}()), dist(-10, 10){
        rows = rSize;
        cols = cSize;
        matSize = rows * cols;

        mat = new T[rows * cols];

        for (int i = 0; i < matSize; i++) {
            mat[i] = dist(rng);//static_cast<T>(dist(rng));
        }
    }

    Matrix(int rSize, int cSize, T arr[]) {
        rows = rSize;
        cols = cSize;
        matSize = rows * cols;

        mat = new T[rows * cols];

        for (int i = 0; i < matSize; i++) {
            mat[i] = arr[i];
        }
    }

    // Default constructor for use in Dataset initialization
    Matrix() {
        rows = -1;
        cols = -1;
        matSize = -1;
        mat = new T[1];
    }

    ~Matrix() {
        mat = nullptr;
    }

    T& get(int row, int col) {
        checkValidIndex(row, col);
        return mat[row * cols + col];
    }

    int getRowSize() {
        return rows;
    }

    int getColSize() {
        return cols;
    }

    void setSize(int rowNum, int colNum) {
        rows = rowNum;
        cols = colNum;
        matSize = rows * cols;
    }

    int getSize() {
        return matSize;
    }

    Matrix copy() {
        Matrix newMat = Matrix(getRowSize(), getColSize(), mat);
        return newMat;
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
        Matrix newMat = copy();

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
        Matrix newMat = copy();

        function mult = [](T x, U y) {return x * y;};
        newMat.updateAll(mult, scalar);
        return newMat;
    }

    template <typename U>
    Matrix operator/(U scalar) {
        // return operator*(1 / scalar);

        Matrix newMat = copy();

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

    // For matrix initialization
    mt19937 rng;
    uniform_real_distribution<double> dist;

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

};

template <typename T>
string to_string(Matrix<T> mat) {
    return mat.toString();
}

// // Functions to allow Dual functionality with Matrix<Dual> objects
// template <typename T>
// Matrix<T> getValue(Matrix<Dual<T>> mat) {
//     Matrix<T> valMat = Matrix<T>(mat.getRowSize(), mat.getColSize());
//
//     for (int r = 0; r < valMat.getRowSize(); r++) {
//         for (int c = 0; c < valMat.getColSize(); c++) {
//             valMat.get(r, c) = mat.get(r, c).getValue();
//         }
//     }
//
//     return valMat;
// }
//
// template <typename T>
// Matrix<T> getDerivative(Matrix<Dual<T>> mat) {
//     Matrix<T> derMat = Matrix<T>(mat.getRowSize(), mat.getColSize());
//
//     for (int r = 0; r < derMat.getRowSize(); r++) {
//         for (int c = 0; c < derMat.getColSize(); c++) {
//             derMat.get(r, c) = mat.get(r, c).getDerivative();
//         }
//     }
//
//     return derMat;
// }

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
    Matrix newMat = mat.copy();

    // function add = [](T x, T y) {return x + y;};
    function customPow = [](T x, U y) {return pow(x, y);};
    newMat.updateAll(customPow, val);
    return newMat;
}

// template <typename T, typename U>
// Matrix<T> pow(Matrix<T> mat, U val) {
//     Matrix newMat = mat.copy();
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