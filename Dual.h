//TODO--implement multi-variable derivatives with array arguments?

#include <iostream>
#include <cmath>
#pragma once
using namespace std;

template <typename T>
class Dual {
public:
    Dual(T value, T derivative) : val(value), der(derivative) {}

    // Default constructor for use in Matrix() declaration
    Dual() : val(0), der(1) {}

    // For use with static_cast<>
    Dual(T val) : val(val), der(1) {}

    T getValue() {
        return val;
    }

    T getDerivative() {
        return der;
    }

    string display() {
        return "(" + to_string(getValue()) + ", " + to_string(getDerivative()) + ")";
    }

    Dual operator()(T val) {
        return Dual(val, 1.0);
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
    T der;  // derivative
};

template <typename T>
string to_string(Dual<T> dual) {
    return to_string(dual.getValue());
}

// Overloaded operators for Dual() objects
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
Dual<T> operator/(T val, Dual<T> dual) {
    return Dual(val / dual.getValue(), -val * dual.getDerivative() / pow(dual.getValue(), 2));
}

// Overloaded mathematical functions for Dual() objects
template <typename T>
Dual<T> pow(Dual<T> dual, T val) {
    return Dual(pow(dual.getValue(), val), val * pow(dual.getValue(), (val - 1)) * dual.getDerivative());
}

template <typename T>
Dual<T> pow(T val, Dual<T> dual) {
    return Dual(pow(val, dual.getValue()), pow(val, dual.getValue()) * log(val) * dual.getDerivative());
}

template <typename T>
Dual<T> log(Dual<T> dual) {
    return Dual(log(dual.getValue()), dual.getDerivative() / dual.getValue());
}

template <typename T>
Dual<T> log10(Dual<T> dual) {
    return Dual(log10(dual.getValue()), dual.getDerivative() / (log(10.0) * dual.getValue()));
}

template <typename T>
Dual<T> log2(Dual<T> dual) {
    return Dual(log10(dual.getValue()), dual.getDerivative() / (log(2.0) * dual.getValue()));
}

template <typename T>
Dual<T> exp(Dual<T> dual) {
    return Dual(exp(dual.getValue()), exp(dual.getValue()) * dual.getDerivative());
}

template <typename T>
Dual<T> sin(Dual<T> dual) {
    return Dual(sin(dual.getValue()), cos(dual.getValue()) * dual.getDerivative());
}

template <typename T>
Dual<T> cos(Dual<T> dual) {
    return Dual(cos(dual.getValue()), -1 * sin(dual.getValue()) * dual.getDerivative());
}


// ADD COMMENTS!!!