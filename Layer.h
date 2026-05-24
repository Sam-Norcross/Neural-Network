#include <iostream>
#include <functional>

#include "Matrix.h"

#pragma once
using namespace std;

template <typename T>
class Layer {
public:
    Layer (int numNodes, int previousNodes, function<Matrix<T>> actFunc) {
        // numNodes is the number of nodes in this layer, previousNodes is the number of nodes in the previous layer,
        // actFunc is the activation function

        nodes = Matrix<T>(numNodes, previousNodes);
        nodes.randomize();

        bias = Matrix<T>(numNodes, 1);
        bias.randomize();

        activation = actFunc;
    }

    Matrix<T> feedForward(Matrix<T> input) {    // TODO--may need to create better vectorized operators so that this process can be done on the entire dataset at once
        Matrix z = matmul(nodes, input) + bias;
        return activation(z);
    }

private:
    Matrix<T> nodes;
    Matrix<T> bias;
    function<Matrix<T>> activation;
};