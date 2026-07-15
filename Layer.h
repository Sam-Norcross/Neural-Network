#include <iostream>
#include <functional>

#include "Matrix.h"

#pragma once
using namespace std;

template <typename T>
class Layer {
public:
    Layer (int numNodes, int previousNodes, Matrix<T> (*actFunc)(Matrix<T>), Matrix<T> (*actDer)(Matrix<T>), double learningRate) :
            numNodes(numNodes), nextLayer(nullptr), previousLayer(nullptr), activation(actFunc), activationDerivative(actDer) {
    // Layer (int numNodes, int previousNodes, function<Matrix<T>(Matrix<T>)> actFunc, function<Matrix<T>(Matrix<T>)> actDer) :
    //         nextLayer(nullptr), previousLayer(nullptr), activation(actFunc), activationDerivative(actDer) {
        // numNodes is the number of nodes in this layer, previousNodes is the number of nodes in the previous layer,
        // actFunc is the activation function, actDer is the derivative of the activation function

        weights = Matrix<T>(numNodes, previousNodes);
        weights.randomize();

        bias = Matrix<T>(numNodes, 1);
        bias.randomize();
    }

    Layer() : numNodes(0), nextLayer(nullptr), previousLayer(nullptr), activation(nullptr), activationDerivative(nullptr) {}

    // Copy constructor
    Layer(const Layer& other) {
        numNodes = other.getNumNodes();
        weights = other.getWeights();
        bias = other.getBias();

        activation = other.getActivation();
        activationDerivative = other.getActivationDerivative();

        nextLayer = other.getNextLayer();
        previousLayer = other.getPreviousLayer();

        aCurrent = other.getA();
        zCurrent = other.getZ();
        deltaCurrent = other.getDelta();
    }

    ~Layer() {
        delete previousLayer;
        delete nextLayer;
    }

    Layer& operator=(const Layer& other) {
        if (this == &other) {
            return *this;
        }

        numNodes = other.getNumNodes();
        weights = other.getWeights();
        bias = other.getBias();

        activation = other.getActivation();
        activationDerivative = other.getActivationDerivative();

        nextLayer = other.getNextLayer();
        previousLayer = other.getPreviousLayer();

        aCurrent = other.getA();
        zCurrent = other.getZ();
        deltaCurrent = other.getDelta();

        return *this;
    }

    // bool operator==(const Layer& other) {    // TODO--probably needed for the copy assignment operator (operator=())
    //
    // }

    Layer* getNextLayer() {
        return nextLayer;
    }

    Layer* getPreviousLayer() {
        return previousLayer;
    }

    void setNextLayer(Layer *next) {
        nextLayer = next;
    }

    void setPreviousLayer(Layer *previous) {
        previousLayer = previous;
    }

    int getNumNodes() {
        return numNodes;
    }



    Matrix<T> feedForward(Matrix<T> input) {
        zCurrent = weights.matMul(input).colAdd(bias);
        aCurrent = activation(zCurrent);
        return aCurrent;
    }

    // TODO
    // Define backpropagate() for a general node and for the input and output nodes?
    // Ex: backpropagate(), backpropagateInput(Matrix<T> input), backpropagateOutput(Matrix<T> output)

    // nodeUpdate and biasUpdate should be the gradient of the matrices
    void backpropagate(double costDerivative, double learningRate) { // Called before backpropagate() for the last layer in the network
        Matrix<T> delta = costDerivative * activationDerivative(zCurrent);
        weights -= learningRate * deltaCurrent.matMul((nextLayer->getA()).transpose());
        bias -= learningRate * deltaCurrent.sumToColVec();
    }

    void backpropagate(Matrix<T> input, double learningRate) {   // Called after backpropagate() for the first layer in the network
        Matrix<T> dCda = (nextLayer->getWeights()).transpose().matMul(nextLayer->getDelta());
        deltaCurrent = dCda * activationDerivative(zCurrent);
        weights -= learningRate * deltaCurrent.matMul(input.transpose());
        bias -= learningRate * deltaCurrent.sumToColVec();
    }

    void backpropagate(Matrix<T> input, double costDerivative, double learningRate) {   // Backpropagation for a network with a single layer
        // TODO--could do this instead of saving zCurrent for each node
        // deltaCurrent = costDerivative * activationDerivative(weights.matMul(input).colAdd(bias));

        deltaCurrent = costDerivative * activationDerivative(zCurrent);
        weights -= learningRate * deltaCurrent.matMul(input.transpose());
        bias -= learningRate * deltaCurrent.sumToColVec();
    }


    void backpropagate(double learningRate) {   // Backpropagation for any general hidden layer
        // Needs weights(n+1), delta(n+1), z(n), a(n-1)

        // dCda3 = (weight4.transpose()).matMul(delta4);
        // Matrix delta3 = dCda3 * reluDerivative(z3);
        // dCdW3 = delta3.matMul(a2.transpose());
        // dCdb3 =  delta3.sumToColVec();

        // Matrix<T> dCda1 = (weight2.transpose()).matMul(delta2);
        // Matrix delta1 = dCda1 * activationDerivative(z1);
        //
        // weights -= delta1.matMul(inputTrain.transpose());
        // bias -= delta1.sumToColVec();

        Matrix<T> dCda;
        dCda = (nextLayer->getWeights()).transpose().matMul(nextLayer->getDelta());

        deltaCurrent = dCda * activationDerivative(zCurrent);

        weights -= learningRate * deltaCurrent.matMul((nextLayer->getA()).transpose());
        bias -= learningRate * deltaCurrent.sumToColVec();
    }

    Matrix<T> getWeights() {
        return weights;
    }

    Matrix<T> getBias() {
        return bias;
    }

    Matrix<T> getA() {
        return aCurrent;
    }

    Matrix<T> getZ() {
        return zCurrent;
    }

    Matrix<T> getDelta() {
        return deltaCurrent;
    }

private:
    int numNodes;
    Matrix<T> weights;
    Matrix<T> bias;
    Matrix<T> (*activation)(Matrix<T>);
    Matrix<T> (*activationDerivative)(Matrix<T>);

    Layer *nextLayer;
    Layer *previousLayer;

    // a = activation(z)
    Matrix<T> aCurrent; // Current pre-activation result of the feed forward process
    Matrix<T> zCurrent; // Current result of the feed forward process from this node
    Matrix<T> deltaCurrent; // Intermediate step for backprop, saved to pass to next layer

protected:
    Matrix<T> getActivation() {
        return activation;
    }

    Matrix<T> getActivationDerivative() {
        return activation;
    }
};







// Activation functions

// Linear activation function
template <typename T>
Matrix<T> linearAct(Matrix<T> input) {
    return input;
}

template <typename T>
Matrix<T> linearActDerivative(Matrix<T> input) {
    return ones<T>(input.getNumRows(), input.getNumCols());
}



// ReLU
template <typename T>
Matrix<T> relu(Matrix<T> input) {
    Matrix<T> newMat = input;

    for (int r = 0; r < input.getNumRows(); r++) {
        for (int c = 0; c < input.getNumCols(); c++) {

            if (newMat.get(r, c) < 0.0) {
                newMat.get(r, c) = static_cast<T>(0.0);
            }

        }
    }

    return newMat;
}

template <typename T>
Matrix<T> reluDerivative(Matrix<T> input) {
    Matrix<T> newMat = input;

    for (int r = 0; r < input.getNumRows(); r++) {
        for (int c = 0; c < input.getNumCols(); c++) {

            if (newMat.get(r, c) <= 0) {
                newMat.get(r, c) = static_cast<T>(0.0);
            } else {
                newMat.get(r, c) = static_cast<T>(1.0);
            }

        }
    }

    return newMat;
}



// ReLU
template <typename T>
Matrix<T> leakyRelu(Matrix<T> input) {
    Matrix<T> newMat = input;

    for (int r = 0; r < input.getNumRows(); r++) {
        for (int c = 0; c < input.getNumCols(); c++) {

            if (newMat.get(r, c) < 0.0) {
                newMat.get(r, c) = 0.1 * input.get(r, c);
            }

        }
    }

    return newMat;
}

template <typename T>
Matrix<T> leakyReluDerivative(Matrix<T> input) {
    Matrix<T> newMat = input;

    for (int r = 0; r < input.getNumRows(); r++) {
        for (int c = 0; c < input.getNumCols(); c++) {

            if (newMat.get(r, c) <= 0) {
                newMat.get(r, c) = static_cast<T>(0.1);
            } else {
                newMat.get(r, c) = static_cast<T>(1.0);
            }

        }
    }

    return newMat;
}



// Sigmoid
template <typename T>
Matrix<T> sigmoid(Matrix<T> input) {
    return 1.0 / (1.0 + exp(-input));
}

template <typename T>
function sigmoidTEST = [](Matrix<T> input) {return sigmoid(input);};

template <typename T>
Matrix<T> sigmoidDerivative(Matrix<T> input) {
    return sigmoid(input) * (1.0 - sigmoid(input));
}








// TODO--old code, designed to run feed forward and backpropagation from previous and next pointers--the new code will
// use these pointers in NeuralNetwork.h to iterate through the list, returning the necessary values

// void feedForward(Matrix<T> input) {
//     Matrix z = weights.matMul(input).colAdd(bias);
//     aCurrent = activation(z);
//     nextLayer->feedForward(aCurrent);
// }
//
// // TODO
// // Define backpropagate() for a general node and for the input and output nodes?
// // Ex: backpropagate(), backpropagateInput(Matrix<T> input), backpropagateOutput(Matrix<T> output)
//
// // nodeUpdate and biasUpdate should be the gradient of the matrices
// void backpropagate() {
//
//     // dCda3 = (weight4.transpose()).matMul(delta4);
//     // Matrix delta3 = dCda3 * reluDerivative(z3);
//     // dCdW3 = delta3.matMul(a2.transpose());
//     // dCdb3 =  delta3.sumToColVec();
//
//     // Matrix<T> dCda1 = (weight2.transpose()).matMul(delta2);
//     // Matrix delta1 = dCda1 * activationDerivative(z1);
//     //
//     // weights -= delta1.matMul(inputTrain.transpose());
//     // bias -= delta1.sumToColVec();
//
//     Matrix<T> dCda;
//     if (nextLayer != nullptr) {
//         dCda = (nextLayer->getWeights()).transpose().matMul(nextLayer->getDelta());
//     } else {
//         // dCda = costDerivative(aCurrent);
//     }
//     deltaCurrent = dCda * activationDerivative(zCurrent);
//
//     if (previousLayer != nullptr) {
//         weights -= deltaCurrent.matMul((previousLayer->getA()).transpose());
//     } else {
//         // weights -= deltaCurrent.matMul(networkInput.transpose());
//     }
//     bias -= deltaCurrent.sumToColVec();
// }