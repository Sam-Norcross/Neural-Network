// Has N layers
// Each layer is composed of node values, bias values, and activation functions
// Has a cost function

// Feed forward: multiplies the matrices from the layers

#include <algorithm>
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

        dataset = Dataset<T>(fileName, depName);

    }

    void partitionDataset(double val) { // val is the fraction of the dataset used for training
        int partitionIndex = static_cast<int> (val * dataset.getNumEntries());
        partitionDataset(partitionIndex);
    }

    void partitionDataset(int partitionIndex) {
        Matrix<double> input = dataset.getData().transpose();
        Matrix<double> output = dataset.getDependent().transpose();

        inputTrain = input.getSlice(0, input.getNumRows(), 0, partitionIndex);
        inputValidate = input.getSlice(0, input.getNumRows(), partitionIndex, input.getNumCols());

        outputTrain = output.getSlice(0, output.getNumRows(), 0, partitionIndex);
        outputValidate = output.getSlice(0, output.getNumRows(), partitionIndex, output.getNumCols());
    }

private:
    Dataset<T> dataset;

    Matrix<T> inputTrain;
    Matrix<T> outputTrain;
    Matrix<T> inputValidate;
    Matrix<T> outputValidate;

    Layer<T> *outputLayer;
    Layer<T> *inputLayer;
};