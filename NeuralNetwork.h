#include <iostream>
#include <string>

#include "Matrix.h"
#include "Layer.h"
#include "Dataset.h"
#include "NeuralNetworkException.h"

#pragma once
using namespace std;



double rmse(Matrix<double> actual, Matrix<double> expected);
double rmseDerivative(Matrix<double> actual, Matrix<double> expected);
double mse(Matrix<double> actual, Matrix<double> expected);
double mseDerivative(Matrix<double> actual, Matrix<double> expected);



template <typename T>
class NeuralNetwork {
public:
    NeuralNetwork(string fileName, string depName, double learningRate, string cost) :
            dataset(Dataset<T>(fileName, depName)), outputLayer(nullptr), inputLayer(nullptr), learningRate(learningRate) {

        dataset = Dataset<T>(fileName, depName);

        if (cost == "RMSE") {
            costFunction = rmse;
            costFunctionDerivative = rmseDerivative;
        }
        else if (cost == "MSE") {
            costFunction = mse;
            costFunction = mseDerivative;
        }

    }

    ~NeuralNetwork() {
        delete inputLayer;
        delete outputLayer;
    }

    void addLayer(int numNodes, string activationType) {
        Layer<T> newLayer;

        // Set activation functions
        if (activationType == "Linear") {
            newLayer = new Layer<T>(numNodes, linearAct, linearActDerivative, learningRate);
        }
        else if (activationType == "ReLU") {
            newLayer = new Layer<T>(numNodes, relu, reluDerivative, learningRate);
        }
        else if (activationType == "Sigmoid") {
            newLayer = new Layer<T>(numNodes, sigmoid, sigmoidDerivative, learningRate);
        } // TODO--add more here
        else {
            throw NeuralNetworkException("Activation function type '" + activationType + "' is unknown.");
        }


        // Reset layer pointers
        if (inputLayer == nullptr) {
            inputLayer = newLayer;
        }

        if (outputLayer == nullptr) {
            outputLayer = newLayer;
        } else {
            outputLayer->setNextLayer(newLayer);
            outputLayer = newLayer;
        }

    }

    void trainNetwork(int epochs) {    // TODO--add functionality for relTol stopping condition in addition to epoch number
        if (inputLayer == nullptr) {
            throw NeuralNetworkException("Network has no layers.");
        }

        // Feed forward--TODO--needs to be inside the loop
        Layer* current = inputLayer;
        Matrix<T> layerOutput = inputTrain;

        while (current != nullptr) {
            layerOutput = current->feedForward(layerOutput);
            current = current->getNextLayer();
        }

        // Backpropagation
        if (inputLayer == outputLayer) {    // If the network has only one layer
            for (int i = 0; i < epochs; i++) {

            }
        }




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

    Matrix<T> (*costFunction)(Matrix<T>);
    Matrix<T> (*costFunctionDerivative)(Matrix<T>);

    Layer<T> *inputLayer;
    Layer<T> *outputLayer;

    double learningRate;
};







// RMSE
double rmse(Matrix<double> actual, Matrix<double> expected) {
    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmse()");
    }

    return sqrt(pow(actual - expected, 2).sum() / expected.getNumCols()); // RMSE
}

Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected) {
    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmseDerivative()");
    }

    return  (actual - expected) / (expected.getNumCols() * rmse(actual, expected)); // RMSE
}



// MSE
double mse(Matrix<double> actual, Matrix<double> expected) {
    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmse()");
    }

    return pow(actual - expected, 2).sum() /  expected.getNumCols(); // MSE
}

Matrix<double> mseDerivative(Matrix<double> actual, Matrix<double> expected) {
    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmseDerivative()");
    }

    return (actual - expected) * 2 / expected.getNumCols(); // MSE
}