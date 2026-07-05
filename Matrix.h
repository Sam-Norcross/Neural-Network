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

    // numRows and numCols are the number of rows and columns, respectively
    Matrix(int numRows, int numCols) : rows(numRows), cols(numCols), matSize(numRows * numCols), mat(new T[numRows * numCols]) {}

    Matrix(int numRows, int numCols, T arr[]) : rows(numRows), cols(numCols), matSize(numRows * numCols), mat(new T[numRows * numCols]) {
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
        Matrix mat(1, getNumCols());
        for (int c = 0; c < getNumCols(); c++) {
            mat.get(0, c) = get(row, c);
        }
        return mat;
    }

    Matrix getCol(int col) {
        Matrix mat(getNumRows(), 1);
        for (int r = 0; r < getNumRows(); r++) {
            mat.get(r, 0) = get(r, col);
        }
        return mat;
    }

    void setRow(int row, Matrix rowMat) {
        if (rowMat.getNumRows() != 1 || rowMat.getNumCols() != getNumCols()) {
            throw MatrixException("Matrix of dimensions" + rowMat.dims() + " cannot be used as a row in a matrix of dimensions " + dims());
        }

        for (int c = 0; c < getNumCols(); c++) {
            get(row, c) = rowMat.get(0, c);
        }
    }

    void setCol(int col, Matrix colMat) {
        if (colMat.getNumCols() != 1 || colMat.getNumRows() != getNumRows()) {
            throw MatrixException("Matrix of dimensions" + colMat.dims() + " cannot be used as a column in a matrix of dimensions " + dims());
        }

        for (int r = 0; r < getNumRows(); r++) {
            get(r, col) = colMat.get(r, 0);
        }
    }

    // Return a region of the matrix as a new matrix
    Matrix getSlice(int startRow, int endRow, int startCol, int endCol) {
        // Ranges do not include the final index (similar to the convention used in Python)

        Matrix newMat(endRow - startRow, endCol - startCol);
        for (int r = startRow; r < endRow; r++) {
            for (int c = startCol; c < endCol; c++) {
                newMat.get(r - startRow, c - startCol) = get(r, c);
            }
        }

        return newMat;
    }

    int getNumRows() {
        return rows;
    }

    int getNumCols() {
        return cols;
    }

    int getSize() {
        return matSize;
    }

    void randomize() {
        randomize(-10.0, 10.0);
    }

    // Assigns random values to all matrix elements
    void randomize(T lowBound, T highBound) {
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
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            return false;
        }
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) != mat2.get(r, c)) {
                    return false;
                }
            }
        }
        return true;
    }

    bool operator<(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            throw MatrixException("Matrices of different dimensions cannot be compared with the < operator.");
        }

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) >= mat2.get(r, c)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool operator>(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            throw MatrixException("Matrices of different dimensions cannot be compared with the > operator.");
        }

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) <= mat2.get(r, c)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool operator<=(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            throw MatrixException("Matrices of different dimensions cannot be compared with the <= operator.");
        }

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) > mat2.get(r, c)) {
                    return false;
                }
            }
        }

        return true;
    }

    bool operator>=(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            throw MatrixException("Matrices of different dimensions cannot be compared with the >= operator.");
        }

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) < mat2.get(r, c)) {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename U>
    bool operator<(U val) {
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) >= val) {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename U>
    bool operator>(U val) {
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) <= val) {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename U>
    bool operator<=(U val) {
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) > val) {
                    return false;
                }
            }
        }

        return true;
    }

    template <typename U>
    bool operator>=(U val) {
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                if (get(r, c) < val) {
                    return false;
                }
            }
        }

        return true;
    }

    Matrix& operator+=(Matrix mat2) {
        *this = *this + mat2;
        return *this;
    }

    Matrix& operator-=(Matrix mat2) {
        *this = *this - mat2;
        return *this;
    }

    Matrix& operator*=(Matrix mat2) {
        *this = *this * mat2;
        return *this;
    }

    Matrix& operator/=(Matrix mat2) {
        *this = *this / mat2;
        return *this;
    }

    Matrix& operator+=(T val) {
        *this = *this + val;
        return *this;
    }

    Matrix& operator-=(T val) {
        *this = *this - val;
        return *this;
    }
    Matrix& operator*=(T val) {
        *this = *this * val;
        return *this;
    }
    Matrix& operator/=(T val) {
        *this = *this / val;
        return *this;
    }

    Matrix operator-() {    // Returns a new matrix that is equivalent to the original matrix * -1
        return -1 * Matrix(*this);
    }

    Matrix operator+(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            string errMsg1 = "Incorrect dimensions: can't add matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getNumRows(), getNumCols());

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                newMat.get(r, c) = get(r, c) + mat2.get(r, c);
            }
        }

        return newMat;
    }

    Matrix operator-(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            string errMsg1 = "Incorrect dimensions: can't subtract matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getNumRows(), getNumCols());

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                newMat.get(r, c) = get(r, c) - mat2.get(r, c);
            }
        }

        return newMat;
    }

    Matrix operator*(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            string errMsg1 = "Incorrect dimensions: can't perform element-wise multiplication on matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getNumRows(), getNumCols());

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                newMat.get(r, c) = get(r, c) * mat2.get(r, c);
            }
        }

        return newMat;
    }

    Matrix operator/(Matrix mat2) {
        if (getNumRows() != mat2.getNumRows() || getNumCols() != mat2.getNumCols()) {
            string errMsg1 = "Incorrect dimensions: can't perform element-wise division on matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getNumRows(), getNumCols());

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                newMat.get(r, c) = get(r, c) / mat2.get(r, c);
            }
        }

        return newMat;
    }

    template <typename U>
    void updateAll(function<T(T, U)> func, U val) {
        // func(T, U) takes in the value in the original matrix and a new value and performs some operation

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                get(r, c) = func(get(r, c), val);
            }
        }
    }

    // Applies a function with one argument to all elements of the matrix
    void map(function<T(T)> func) {
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                get(r, c) = func(get(r, c));
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
        if (getNumCols() != mat2.getNumRows()) {
            string errMsg1 = "Incorrect dimensions: can't multiply matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException( errMsg1 + dims() + errMsg2 + mat2.dims());
        }

        Matrix newMat = Matrix(getNumRows(), mat2.getNumCols());

        double dotProduct = 0;
        for (int r1 = 0; r1 < getNumRows(); r1++) {     // Iterate over the rows of mat1
            for (int c2 = 0; c2 < mat2.getNumCols(); c2++) {    // Iterate over the cols of mat2
                for (int c1 = 0; c1 < getNumCols(); c1 ++) {    // Iterate over the cols of mat1/the rows of mat2
                    dotProduct += get(r1, c1) * mat2.get(c1, c2);
                }
                newMat.get(r1, c2) = dotProduct;
                dotProduct = 0;
            }
        }

        return newMat;
    }

    // Adds each column of the matrix to colVec (a column vector), returns a matrix of the same size
    Matrix colAdd(Matrix colVec) {
        if (getNumRows() != colVec.getNumRows()) {
            string errMsg1 = "Incorrect dimensions: can't add columns of matrices with dimensions ";
            string errMsg2 = " and ";
            throw MatrixException(errMsg1 + dims() + errMsg2 + colVec.dims());
        }

        Matrix newMat = Matrix(getNumRows(), getNumCols());

        for (int c = 0; c < getNumCols(); c++) {
            Matrix newCol = getCol(c) + colVec;

            for (int r = 0; r < getNumRows(); r++) {
                newMat.get(r, c) = newCol.get(r, 0);
            }
        }

        return newMat;
    }

    T sum() {
        T sumVal = 0;
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                sumVal += get(r, c);
            }
        }

        return sumVal;
    }

    // Sum rows, producing a column vector TODO--replace with sum() with an axis argument?
    Matrix sumToColVec() {
        Matrix newMat = Matrix(getNumRows(), 1);

        int rowSum = 0;
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                rowSum += get(r, c);
            }
            newMat.get(r, 0) = rowSum;
            rowSum = 0;
        }

        return newMat;
    }

    Matrix transpose() {
        Matrix newMat = Matrix(getNumCols(), getNumRows());

        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                newMat.get(c, r) = get(r, c);
            }
        }

        return newMat;
    }



    string toString() {
        string matString = "[";
        for (int r = 0; r < getNumRows(); r++) {
            for (int c = 0; c < getNumCols(); c++) {
                matString += to_string(get(r, c));
                if (c < getNumCols() - 1) {
                    matString += ", ";
                } else if (r < getNumRows() - 1) {
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

// protected:
//
//     void setSize(int rowNum, int colNum) {
//         rows = rowNum;
//         cols = colNum;
//         matSize = rows * cols;
//
//         delete [] mat;
//         mat = new T[rows * cols];
//     }

};

template <typename T>
Matrix<T> fill(int rows, int cols, T val) {
    Matrix<T> newMat = Matrix<T>(rows, cols);
    for (int r = 0; r < newMat.getNumRows(); r++) {
        for (int c = 0; c < newMat.getNumCols(); c++) {
            newMat.get(r, c) = val;
        }
    }

    return newMat;
}

template <typename T>
Matrix<T> zeros(int rows, int cols) {
    return fill(rows, cols, static_cast<T>(0));
}

template <typename T>
Matrix<T> ones(int rows, int cols) {
    return fill(rows, cols, static_cast<T>(1));
}

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
Matrix<T> operator-(T scalar, Matrix<U> mat) {
    Matrix newMat = mat;

    function customSubtract = [](T x, T y) {return y - x;};
    newMat.updateAll(customSubtract, scalar);
    return newMat;
}

template <typename T, typename U>
Matrix<T> operator*(U scalar, Matrix<T> mat) {
    return mat * scalar;
}

template <typename T, typename U>
Matrix<T> operator/(U scalar, Matrix<T> mat) {
    function customDiv = [=](T x) {return scalar / x;};

    mat.map(customDiv);
    return mat;
}

// Overloaded mathematical functions for Matrix() objects
template <typename T, typename U>
Matrix<T> pow(Matrix<T> mat, U exponent) {
    Matrix newMat = mat;

    function customPow = [=](T x) {return pow(x, exponent);};
    newMat.map(customPow);
    return newMat;
}

template <typename T>
Matrix<T> exp(Matrix<T> mat) {
    function expFunc = [](T x) {return exp(x);};
    mat.map(expFunc);
    return mat;
}

// TODO--implement log, log10, log2, exp, sin, cos
// TODO--change updateAll() to map() that applies the function directly to each value without needing an extra argument?