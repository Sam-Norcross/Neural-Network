// Has N layers
// Each layer is composed of node values, bias values, and activation functions
// Has a cost function

// Feed forward: multiplies the matrices from the layers

#include <iostream>

#include "Matrix.h"
#include "Layer.h"
#include "Dataset.h"

#pragma once
using namespace std;


template <typename T>
class NeuralNetwork {
public:
    NeuralNetwork(string fileName, string depName) : data(Dataset<T>(fileName, depName)), outputLayer(nullptr), inputLayer(nullptr) {

    }

private:
    Dataset<T> data;

    Layer<T> *outputLayer;
    Layer<T> *inputLayer;

};