#include <iostream>
#include <string>

#include "Matrix.h"
#include "Layer.h"
#include "Dataset.h"
#include "NeuralNetworkException.h"

#pragma once
using namespace std;



double rmse(Matrix<double> actual, Matrix<double> expected);
Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected);
double mse(Matrix<double> actual, Matrix<double> expected);
Matrix<double> mseDerivative(Matrix<double> actual, Matrix<double> expected);

// TODO--add function to make predictions with trained dataset--maybe make feedForwardFull a private method and create a wrapper for predictions?
// TODO--add functionality to save the trained NN to a file so it can be loaded and used for predictions without retraining
// TODO--add functionality so that the learning rate can be adjusted as the model trains to increase speed?


template <typename T>
class NeuralNetwork {
public:
    NeuralNetwork(string fileName, string depName, double learningRate, string cost) :
            dataset(Dataset<T>(fileName, depName)), numLayers(0), outputLayer(nullptr), inputLayer(nullptr),
            learningRate(learningRate), dataPartitioned(false) {

        dataset = Dataset<T>(fileName, depName);

        if (cost == "RMSE") {
            costFunction = rmse;
            costFunctionDerivative = rmseDerivative;
        }
        else if (cost == "MSE") {
            costFunction = mse;
            costFunctionDerivative = mseDerivative;
        }

    }

    ~NeuralNetwork() {
        // delete inputLayer;
        // delete outputLayer;
    }

    void addLayer(int numNodes, string activationType) {
        Layer<T> *newLayer;

        int prevNodes;
        if (outputLayer == nullptr) {
            prevNodes = dataset.getNumFields();
        } else {
            prevNodes = outputLayer->getNumNodes();
        }



        // Set activation functions
        if (activationType == "Linear") {
            newLayer = new Layer<T>(numNodes, prevNodes, linearAct, linearActDerivative);
        }
        else if (activationType == "ReLU") {
            newLayer = new Layer<T>(numNodes, prevNodes, relu, reluDerivative);
        }
        else if (activationType == "Sigmoid") {
            newLayer = new Layer<T>(numNodes, prevNodes, sigmoid, sigmoidDerivative);
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
            newLayer->setPreviousLayer(outputLayer);
            outputLayer = newLayer;
        }

        numLayers++;
    }

    Matrix<T> feedForwardFull(Matrix<T> input) {
        Layer<T> *current = inputLayer;
        Matrix<T> currentOutput = input;

        while (current != nullptr) {
            currentOutput = current->feedForward(currentOutput);
            current = current->getNextLayer();
        }

        return currentOutput;
    }

    void backpropagateFull(Matrix<T> input, Matrix<T> output, Matrix<T> expected) {
        // TODO--maybe have this function return another function so it doesn't need to check the number of layers each time?
        if (numLayers == 1) {   // If the network only has one layer
            inputLayer->backpropagateSingleLayer(input, costFunctionDerivative(output, expected), learningRate);
        } else {

            // backpropagateLastLayer(Matrix<T> costDerivative, double learningRate)
            // backpropagate(double learningRate)
            // backpropagateFirstLayer(Matrix<T> input, double learningRate)

            outputLayer->backpropagateLastLayer(costFunctionDerivative(output, expected), learningRate);

            Layer<T> *current = outputLayer->getPreviousLayer();
            while (current != inputLayer) {
                current->backpropagate(learningRate);

                current = current->getPreviousLayer();
            }

            inputLayer->backpropagateFirstLayer(input, learningRate);
        }
    }

    void trainNetwork(int epochs) {    // TODO--add functionality for relTol stopping condition in addition to epoch number
        if (numLayers == 0) {
            throw NeuralNetworkException("Network has no layers.");
        }

        checkDataPartitioned();    // Throw an error if the dataset is not partitioned into training and validation data.


        Matrix<T> output;
        double cost;

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++) {

            // Feed forward
            output = feedForwardFull(inputTrain);
            cost = costFunction(output, outputTrain);
            cout << cost << endl;


            // Backpropagation
            backpropagateFull(inputTrain, output, outputTrain);

        }


        // Validate data
        output = feedForwardFull(inputValidate);
        cost = costFunction(output, outputValidate);

        cout << "The cost of the validation data is " << cost << endl;




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

        dataPartitioned = true;
    }

    Matrix<T> getTrainingInput() {
        checkDataPartitioned();

        return inputTrain;
    }

    Matrix<T> getTrainingOutput() {
        checkDataPartitioned();

        return outputTrain;
    }

    Matrix<T> getValidationInput() {
        checkDataPartitioned();

        return inputValidate;
    }

    Matrix<T> getValidationOutput() {
        checkDataPartitioned();

        return outputValidate;
    }

private:
    Dataset<T> dataset;

    bool dataPartitioned;
    // Keeps track of whether the dataset has been partitioned or not (if inputTrain, outputTrain, etc. have been declared)
    Matrix<T> inputTrain;
    Matrix<T> outputTrain;
    Matrix<T> inputValidate;
    Matrix<T> outputValidate;

    T (*costFunction)(Matrix<T>, Matrix<T>);
    Matrix<T> (*costFunctionDerivative)(Matrix<T>, Matrix<T>);

    int numLayers;
    Layer<T> *inputLayer;
    Layer<T> *outputLayer;

    double learningRate;



    void checkDataPartitioned() {
        if (!dataPartitioned) {
            throw NeuralNetworkException("Training and validation data are not properly partitioned. Be sure to call 'partitionDataset' before training network.");
        }
    }
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