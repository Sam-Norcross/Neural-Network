#include <iostream>
#include "Matrix.h"
#pragma once
using namespace std;

template <typename T>
class Dual {
public:
    Dual(T value, T derivative) : val(value), der(derivative) {}

    T getValue() {
        return val;
    }

    T getDerivative() {
        return der;
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

    // Dual operator/(Dual dual2) {
    //     return Dual(getValue() / dual2.getValue(), );
    // }

    Dual operator/(T val) {
        return Dual(getValue() / val, getDerivative() / val);
    }

private:
    T val;  // value
    T der;  // derivative
};

template <typename T>
Dual<T> operator+(T val, Dual<T> dual) {
    return dual + val;
}

template <typename T>
Dual<T> operator-(T val, Dual<T> dual) {
    return Dual(val - dual.getValue(), dual.getDerivative());;
}

template <typename T>
Dual<T> operator*(T val, Dual<T> dual) {
    return dual * val;
}

template <typename T>
Dual<T> pow(Dual<T> dual, T val) {
    return Dual(pow(dual.getValue(), val), val * pow(dual.getValue(), (val - 1)) * dual.getDerivative());
}