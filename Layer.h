#include <iostream>
#include <functional>

#include "Matrix.h"
#include "Dual.h"

#pragma once
using namespace std;

template <typename T>
class Layer {
public:
    Layer (int numNodes, int previousNodes, function<Matrix<Dual<T, T>>(Matrix<Dual<T, T>>)> actFunc) {
        // numNodes is the number of nodes in this layer, previousNodes is the number of nodes in the previous layer,
        // actFunc is the activation function

        nodes = Matrix<Dual<T, T>>(numNodes, previousNodes);
        nodes.randomize();

        bias = Matrix<Dual<T, T>>(numNodes, 1);
        bias.randomize();

        activation = actFunc;
    }

    Matrix<Dual<T, T>> feedForward(Matrix<T> input) {    // TODO--may need to create better vectorized operators so that this process can be done on the entire dataset at once
        Matrix z = matmul(nodes, input) + bias;
        return activation(z);
    }

    // nodeUpdate and biasUpdate should be the gradient of the matrices
    void update(Matrix<Dual<T, T>> nodeUpdate, Matrix<Dual<T, T>> biasUpdate) {
        nodes -= nodeUpdate;
        bias -= biasUpdate;
    }

private:
    Matrix<T> nodes;
    Matrix<T> bias;
    function<Matrix<T>> activation;
};

// Activation functions
// template <typename T>
// T NAME(T x) {
//
// }


// TODO--comparison operators need to be defined for both Matrix and Dual classes
// TODO--Matrix.h may need better vectorization to correctly handle multiple data points at once
template <typename T>
T relu(T x) {
    if (x > 0) {
        return x;
    }

    return 0;
}

template <typename T>
Matrix<T> relu(Matrix<T> mat) {
    mat.map([](T x) {return relu(x);});
    return mat;
}


// TODO--this maybe allows for the templates to capture Matrix<Dual<T, T>> objects?
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