#include <iostream>
#include <functional>
#include <nlohmann/json.hpp>

#include "Matrix.h"
#include "LayerException.h"

#pragma once
using namespace std;



template <typename T> Matrix<T> linearAct(Matrix<T> input);
template <typename T> Matrix<T> linearActDerivative(Matrix<T> input);
template <typename T> Matrix<T> relu(Matrix<T> input);
template <typename T> Matrix<T> reluDerivative(Matrix<T> input);
template <typename T> Matrix<T> leakyRelu(Matrix<T> input);
template <typename T> Matrix<T> leakyReluDerivative(Matrix<T> input);
template <typename T> Matrix<T> sigmoid(Matrix<T> input);
template <typename T> Matrix<T> sigmoidDerivative(Matrix<T> input);



template <typename T>
class Layer {
public:
    Layer (int numNodes, int previousNodes, const string& activationName) :
            numNodes(numNodes), prevNumNodes(previousNodes), nextLayer(nullptr), previousLayer(nullptr),
            activationType(activationName) {
        // numNodes is the number of nodes in this layer, previousNodes is the number of nodes in the previous layer,
        // actFunc is the activation function, actDer is the derivative of the activation function

        weights = Matrix<T>(numNodes, previousNodes);
        weights.randomize();//-.1, .1);    // TODO--add another constructor that has limits for randomized values?
        // TODO--what is the best way to initialize the weights?

        bias = zeros<T>(numNodes, 1);   // Setting biases to 0 is the standard for bias initialization


        // Set activation functions
        if (activationType == "Linear") {
            activation = linearAct;
            activationDerivative = linearActDerivative;
        }
        else if (activationType == "ReLU") {
            activation = relu;
            activationDerivative = reluDerivative;
        }
        else if (activationType == "Leaky ReLU") {
            activation = leakyRelu;
            activationDerivative = leakyReluDerivative;
        }
        else if (activationType == "Sigmoid") {
            activation = sigmoid;
            activationDerivative = sigmoidDerivative;
        } // TODO--add more here
        else {
            throw LayerException("Activation function type '" + activationType + "' is unknown.");
        }
    }

    Layer() : numNodes(0), prevNumNodes(0), nextLayer(nullptr), previousLayer(nullptr),
                activationType("TEST--Layer() constructor"), activation(nullptr), activationDerivative(nullptr) {}

    // Copy constructor
    Layer(const Layer& other) {
        numNodes = other.numNodes;
        prevNumNodes = other.prevNumNodes;
        weights = other.weights;
        bias = other.bias;


        activationType = other.activationType;
        activation = other.activation;
        activationDerivative = other.activationDerivative;

        nextLayer = other.nextLayer;
        previousLayer = other.previousLayer;

        // TODO--not needed?
        // aCurrent = other.aCurrent;
        // zCurrent = other.zCurrent;
        // deltaCurrent = other.deltaCurrent;
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
        prevNumNodes = other.getPrevNumNodes();
        weights = other.getWeights();
        bias = other.getBias();

        activationType = other.getActivationType();
        activation = other.getActivation();
        activationDerivative = other.getActivationDerivative();

        nextLayer = other.getNextLayer();
        previousLayer = other.getPreviousLayer();

        // TODO--probably not needed?
        // aCurrent = other.getA();
        // zCurrent = other.getZ();
        // deltaCurrent = other.getDelta();

        return *this;
    }

    // Layer equality only checks the structure of the nodes, not the matrices, as they are randomized
    // bool operator==(const Layer& other) {
    bool operator==(const Layer& other) const {

        if (getNumNodes() != other.getNumNodes()) {
            return false;
        }

        if (getPrevNumNodes() != other.getPrevNumNodes()) {
            return false;
        }

        if (getActivationType() != other.getActivationType()) {
            return false;
        }

        // if (getNextLayer() != other.getNextLayer()) {
        //     return false;
        // }
        //
        // if (getPreviousLayer() != other.getPreviousLayer()) {
        //     return false;
        // }

        // TODO--aCurrent, zCurrent, and deltaCurrent don't need to be checked because they are only stored temporarily
        // // a = activation(z)
        // Matrix<T> aCurrent; // Current pre-activation result of the feed forward process
        // Matrix<T> zCurrent; // Current result of the feed forward process from this node
        // Matrix<T> deltaCurrent; // Intermediate step for backprop, saved to pass to next layer

        return true;
    }

    Layer* getNextLayer() const {
        return nextLayer;
    }

    Layer* getPreviousLayer() const {
        return previousLayer;
    }

    void setNextLayer(Layer *next) {
        nextLayer = next;
    }

    void setPreviousLayer(Layer *previous) {
        previousLayer = previous;
    }



    Matrix<T> feedForward(Matrix<T> input) {
        zCurrent = weights.matMul(input).colAdd(bias);
        aCurrent = activation(zCurrent);
        return aCurrent;
    }

    void backpropagateLastLayer(Matrix<T> costDerivative) { // Called before backpropagate() for the last layer in the network
        deltaCurrent = costDerivative * activationDerivative(zCurrent);
        // deltaCurrent = costDerivative * nextLayer->getActivationDerivative()(zCurrent); // TODO

        weightGradient = deltaCurrent.matMul(previousLayer->getA().transpose());
        biasGradient = deltaCurrent.sumToColVec();
    }

    void backpropagateFirstLayer(Matrix<T> input) {   // Called after backpropagate() for the first layer in the network

        // cout << "First layer:" << endl;
        // cout << "Weights: ";
        // weights.display();
        // cout << "Bias: ";
        // bias.display();
        // cout << "zCurrent: ";
        // zCurrent.display();
        // cout << endl;

        Matrix<T> dCda = (nextLayer->getWeights()).transpose().matMul(nextLayer->getDelta());
        deltaCurrent = dCda * activationDerivative(zCurrent);
        // deltaCurrent = dCda * nextLayer->getActivationDerivative()(zCurrent); // TODO

        weightGradient = deltaCurrent.matMul(input.transpose());
        biasGradient = deltaCurrent.sumToColVec();
        // weights -= learningRate * deltaCurrent.matMul(input.transpose());
        // bias -= learningRate * deltaCurrent.sumToColVec();


        // cout << "Weight gradient: ";
        // deltaCurrent.matMul(input.transpose()).display();
        // cout << "Bias gradient: ";
        // deltaCurrent.sumToColVec().display();
        // cout << endl;
        //
        // cout << "deltaCurrent: ";
        // deltaCurrent.display();
        // cout << "input: ";
        // input.display();
        //
        // cout << endl << "--------------" << endl << endl;
    }

    void backpropagateSingleLayer(Matrix<T> input, Matrix<T> costDerivative) {   // Backpropagation for a network with a single layer

        deltaCurrent = costDerivative * activationDerivative(zCurrent);
        // deltaCurrent = costDerivative * nextLayer->getActivationDerivative(zCurrent); // TODO--is this correct? Maybe it should be previousLayer?
        weightGradient = deltaCurrent.matMul(input.transpose());
        biasGradient  = deltaCurrent.sumToColVec();
    }

    void backpropagate() {   // Backpropagation for any general hidden layer
        Matrix<T> dCda = (nextLayer->getWeights()).transpose().matMul(nextLayer->getDelta());

        deltaCurrent = dCda * activationDerivative(zCurrent);
        // deltaCurrent = dCda * nextLayer->getActivationDerivative()(zCurrent); // TODO

        weightGradient = deltaCurrent.matMul((previousLayer->getA()).transpose());
        biasGradient = deltaCurrent.sumToColVec();
    }

    void updateLayer(double learningRate) {
        weights -= learningRate * weightGradient;
        bias -= learningRate * biasGradient;

        // cout << "Weight gradient:\n";
        // weightGradient.display();
        // cout << "Bias gradient:\n";
        // biasGradient.display();
    }

    int getNumNodes() const {
        return numNodes;
    }

    int getPrevNumNodes() const {
        return prevNumNodes;
    }

    Matrix<T> getWeights() const {
        return weights;
    }

    Matrix<T> getBias() const {
        return bias;
    }

    void setWeights(Matrix<T> weightsMat) {
        if (weights.getNumRows() != weightsMat.getNumRows() || weights.getNumCols() != weightsMat.getNumCols()) {
            throw LayerException("Cannot set Layer weights with dimensions " + weights.dims() + " to have dimensions " + weightsMat.dims() + ".");
        }
        weights = weightsMat;
    }

    void setBias(Matrix<T> biasMat) {
        if (bias.getNumRows() != biasMat.getNumRows() || bias.getNumCols() != biasMat.getNumCols()) {
            throw LayerException("Cannot set Layer weights with dimensions " + bias.dims() + " to have dimensions " + biasMat.dims() + ".");
        }
        bias = biasMat;
    }

    string getActivationType() const {
        return activationType;
    }

    Matrix<T> getA() const {
        return aCurrent;
    }

    Matrix<T> getZ() const {
        return zCurrent;
    }

    Matrix<T> getDelta() const {
        return deltaCurrent;
    }

    void randomizeLayer(T lowBound, T highBound) {
        weights.randomize(lowBound, highBound);
        bias.randomize(lowBound, highBound);
    }

private:
    int numNodes;
    int prevNumNodes;
    Matrix<T> weights;
    Matrix<T> bias;

    string activationType;
    Matrix<T> (*activation)(Matrix<T>);
    Matrix<T> (*activationDerivative)(Matrix<T>);

    Layer *nextLayer;
    Layer *previousLayer;

    // a = activation(z)
    Matrix<T> aCurrent; // Current pre-activation result of the feed forward process
    Matrix<T> zCurrent; // Current result of the feed forward process from this node
    Matrix<T> deltaCurrent; // Intermediate step for backprop, saved to pass to next layer

    Matrix<T> weightGradient;
    Matrix<T> biasGradient;

protected:
    // Matrix<T> getActivation() const {
    //     return activation;
    // }

    // Matrix<T> getActivationDerivative() const {
    //     return activationDerivative();
    // }

    Matrix<T> (*getActivation() const)(Matrix<T>) {
        return activation;
    }

    Matrix<T> (*getActivationDerivative() const)(Matrix<T>) {
        return activationDerivative;
    }
};



// JSON serialization and deserialization functions
template <typename T>
void to_json(nlohmann::json& j, const Layer<T>& layer) {
    j["numNodes"] = layer.getNumNodes();
    j["prevNumNodes"] = layer.getPrevNumNodes();
    j["weights"] = layer.getWeights();
    j["bias"] = layer.getBias();

    j["activationType"] = layer.getActivationType();
}

template <typename T>
void from_json(const nlohmann::json& j, Layer<T>& layer) {
    layer = Layer<T>(j["numNodes"], j["prevNumNodes"], j["activationType"]);

    layer.setWeights(j["weights"]);
    layer.setBias(j["bias"]);

}



template <typename T>
string to_string(Layer<T> layer) {
    string layerString = "";
    layerString += "numNodes: " + to_string(layer.getNumNodes()) + "\n";
    layerString += "activationType: " + layer.getActivationType() + "\n";
    layerString += "weights:\n" + to_string(layer.getWeights()) + "\n";
    layerString += "bias:\n" + to_string(layer.getBias());

    return layerString;
}




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
                newMat.get(r, c) = 0.0;//static_cast<T>(0.0);
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
                newMat.get(r, c) = 0.0;//static_cast<T>(0.0);
            } else {
                newMat.get(r, c) = 1.0;//static_cast<T>(1.0);
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
Matrix<T> sigmoidDerivative(Matrix<T> input) {
    return sigmoid(input) * (1.0 - sigmoid(input));
}
