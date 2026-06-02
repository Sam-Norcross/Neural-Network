//TODO--implement multi-variable derivatives with array arguments?

#include <iostream>
#include <cmath>

#include "Matrix.h"
#pragma once
using namespace std;

template <typename T, typename U>
class Dual {
public:
    Dual(T value, U derivative) : val(value), der(derivative) {}
    // Dual(T value, T derivative) {
    //     val = value;
    //     der = Matrix(1, 1, {derivative});
    // }

    // Dual(T value, T derivative[], int rows, int cols) {
    //     val = value;
    //     der = Matrix(rows, cols, {derivative});
    // }

    // Default constructor for use in Matrix() declaration
    Dual() : val(0.0), der(1.0) {}
    // Dual() : val(0.0), der(Matrix<T>(1, 1, {1.0})) {}

    // For use with static_cast<>
    Dual(T val) : val(val), der(1.0) {}
    // Dual(T val) : val(val), der(Matrix<T>(1, 1, {1.0})){}

    T getValue() {
        return val;
    }

    U getDerivative() {
        return der;
    }

    void display() {
        cout << "(" + to_string(getValue()) + ", " + to_string(getDerivative()) + ")" << endl;
    }

    bool operator==(Dual dual2) {
        return (getValue() == dual2.getValue()) && (getDerivative() == dual2.getDerivative());
    }

    bool operator!=(Dual dual2) {
        return !operator==(dual2);
    }

    Dual operator+(Dual dual2) {
        return Dual(getValue() + dual2.getValue(), getDerivative() + dual2.getDerivative());
    }

    Dual operator+(T val) {
        return Dual(getValue() + val, getDerivative());
    }

    Dual operator-(Dual dual2) {
        return Dual(getValue() - dual2.getValue(), getDerivative() - dual2.getDerivative());
    }

    Dual operator-(T val) {
        return Dual(getValue() - val, getDerivative());
    }

    Dual operator*(Dual dual2) {
        return Dual(getValue() * dual2.getValue(), getValue() * dual2.getDerivative() + dual2.getValue() * getDerivative());
    }

    Dual operator*(T val) {
        return Dual(getValue() * val, getDerivative() * val);
    }

    Dual operator/(Dual dual2) {
        return Dual(getValue() / dual2.getValue(), getDerivative() / dual2.getValue() - (getValue() * dual2.getDerivative()) / pow(dual2.getValue(), 2));
    }

    Dual operator/(T val) {
        return Dual(getValue() / val, getDerivative() / val);
    }

private:
    T val;  // value
    U der;  // derivative
    // Matrix<T> der;  // derivative
};

// Deduction guides--TODO--can this remove the need for <double, double> in test case "Dual cast to double"?
// template <typename  T, typename U>
// Dual(T, U) -> Dual<T, U>;

template <typename T, typename U>
string to_string(Dual<T, U> dual) {
    return to_string(dual.getValue());
}


// Overloaded operators for Dual() objects
template <typename T, typename U>
Dual<T, U> operator+(T val, Dual<T, U> dual) {
    return dual + val;
}

template <typename T, typename U>
Dual<T, U> operator-(T val, Dual<T, U> dual) {
    return Dual(val - dual.getValue(), dual.getDerivative());;
}

template <typename T, typename U>
Dual<T, U> operator*(T val, Dual<T, U> dual) {
    return dual * val;
}

template <typename T, typename U>
Dual<T, U> operator/(T val, Dual<T, U> dual) {
    return Dual(val / dual.getValue(), -val * dual.getDerivative() / pow(dual.getValue(), 2));
}

// Overloaded mathematical functions for Dual() objects
template <typename T, typename U>
Dual<T, U> pow(Dual<T, U> dual, T val) {
    return Dual(pow(dual.getValue(), val), val * pow(dual.getValue(), (val - 1)) * dual.getDerivative());
}

template <typename T, typename U>
Dual<T, U> pow(T val, Dual<T, U> dual) {
    return Dual(pow(val, dual.getValue()), pow(val, dual.getValue()) * log(val) * dual.getDerivative());
}

template <typename T, typename U>
Dual<T, U> log(Dual<T, U> dual) {
    return Dual(log(dual.getValue()), dual.getDerivative() / dual.getValue());
}

template <typename T, typename U>
Dual<T, U> log10(Dual<T, U> dual) {
    return Dual(log10(dual.getValue()), dual.getDerivative() / (log(10.0) * dual.getValue()));
}

template <typename T, typename U>
Dual<T, U> log2(Dual<T, U> dual) {
    return Dual(log10(dual.getValue()), dual.getDerivative() / (log(2.0) * dual.getValue()));
}

template <typename T, typename U>
Dual<T, U> exp(Dual<T, U> dual) {
    return Dual(exp(dual.getValue()), exp(dual.getValue()) * dual.getDerivative());
}

template <typename T, typename U>
Dual<T, U> sin(Dual<T, U> dual) {
    return Dual(sin(dual.getValue()), cos(dual.getValue()) * dual.getDerivative());
}

template <typename T, typename U>
Dual<T, U> cos(Dual<T, U> dual) {
    return Dual(cos(dual.getValue()), -1 * sin(dual.getValue()) * dual.getDerivative());
}



// Functions to allow Dual functionality with Matrix<Dual> objects
template <typename T, typename U>
Matrix<T> getValue(Matrix<Dual<T, U>> mat) {
    Matrix<T> valMat = Matrix<T>(mat.getNumRows(), mat.getNumCols());

    for (int r = 0; r < valMat.getNumRows(); r++) {
        for (int c = 0; c < valMat.getNumCols(); c++) {
            valMat.get(r, c) = mat.get(r, c).getValue();
        }
    }

    return valMat;
}

template <typename T, typename U>
Matrix<U> getDerivative(Matrix<Dual<T, U>> mat) {
    Matrix<U> derMat = Matrix<U>(mat.getNumRows(), mat.getNumCols());

    for (int r = 0; r < derMat.getNumRows(); r++) {
        for (int c = 0; c < derMat.getNumCols(); c++) {
            derMat.get(r, c) = mat.get(r, c).getDerivative();
        }
    }

    return derMat;
}


// ADD COMMENTS!!!