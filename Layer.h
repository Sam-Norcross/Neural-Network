#include <iostream>
#include <functional>

#include "Matrix.h"

#pragma once
using namespace std;

template <typename T>
class Layer {
public:
    Layer (int numNodes, int previousNodes, function<Matrix<T>(Matrix<T>)> actFunc) {
        // numNodes is the number of nodes in this layer, previousNodes is the number of nodes in the previous layer,
        // actFunc is the activation function

        weights = Matrix<T>(numNodes, previousNodes);
        weights.randomize();

        bias = Matrix<T>(numNodes, 1);
        bias.randomize();

        activation = actFunc;
    }

    Matrix<T> feedForward(Matrix<T> input) {    // TODO--may need to create better vectorized operators so that this process can be done on the entire dataset at once
        // Matrix z = matmul(weights, input) + bias;

        Matrix z = weights.matMul(input).colAdd(bias);
        return activation(z);
    }

    // nodeUpdate and biasUpdate should be the gradient of the matrices
    void backpropagate(Matrix<T> nodeUpdate, Matrix<T> biasUpdate) {
        weights -= nodeUpdate;
        bias -= biasUpdate;
    }

private:
    Matrix<T> weights;
    Matrix<T> bias;
    function<Matrix<T>(Matrix<T>)> activation;
};

// Activation functions
// template <typename T>
// T NAME(T x) {
//
// }


// TODO--comparison operators need to be defined for both Matrix and Dual classes
// TODO--Matrix.h may need better vectorization to correctly handle multiple data points at once
template <typename T, typename U>
Dual<T, U> reluSingle(Dual<T, U> x) {
    if (x > 0) {
        return x;
    }

    return Dual<T, U>(0, 0);
}

// template <typename T>
// T relu(T x) {
//     if (x > 0) {
//         return x;
//     }
//
//     return 0;
// }

// template <typename T>
// Matrix<T> relu(Matrix<T> mat) {
//     mat.map([](T x) {return relu(x);});
//     return mat;
// }

// template <typename T, typename U>
// Dual<T, U> reluF(Dual<T, U> x) {
//     if (x > 0) {
//         return x;
//     }
//
//     return Dual<T, U>(0, 0);
// }

// template <typename T, typename U>
// function<Matrix<Dual<T, U>>(Matrix<Dual<T, U>>)>  relu() {
//     function reluFunc [] (Matrix<Dual<T, U>> x) {return x.map(reluF(x));};
//     return reluFunc;
// }

// Works
// function<Matrix<Dual<double, double>>(Matrix<Dual<double, double>>)>  relu() {
//     // Dual<double, double> arr[4] = {Dual<double, double>(0), Dual<double, double>(-1), Dual<double, double>(-2), Dual<double, double>(3)};
//     // Matrix mat(2, 2, arr);
//
//     function reluFunc = [] (Matrix<Dual<double, double>> x) {return x;};
//     return reluFunc;
// }

// Works
function relu = [] (Matrix<Dual<double, double>> x) {return x;};

// auto relu = [] (Matrix<Dual<double, double>> x) {return x.map(reluSingle);};
// TODO--maybe have Layer() accept a function with signature <Dual<T, U>(Dual<T, U>)> and then apply map() when it is called?
// TODO--map() might need to be adjusted to return a new matrix instead of modifying the matrix in place


// TODO--this maybe allows for the templates to capture Matrix<Dual<T, T>> objects? (outdated--needs to accept Matrix<Dual<T, U>>)
// TODO--there might be a better way to do it
template <typename T>
using dualMatrix = Matrix<Dual<T, T>>;

template <typename T>
T sigmoid(T x) {
    return 1 / (1 + exp(-x));
}

template <typename T>
Matrix<T> sigmoid(Matrix<T> mat) {
    mat.map([](T x) {return sigmoid(x);});
    return mat;
}