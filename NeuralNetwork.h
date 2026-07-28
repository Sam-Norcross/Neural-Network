#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

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
    // NeuralNetwork(string fileName, string depName, double learningRate, string cost) :
    //         datasetFilePath(fileName), dataset(Dataset<T>(fileName, depName)), numLayers(0),
    //         outputLayer(nullptr), inputLayer(nullptr), learningRate(learningRate), dataPartitioned(false), costType(cost) {
    //
    //     dataset = Dataset<T>(fileName, depName); // TODO--already handled in initializer list?
    //
    //     setCostFunction(cost);
    // }

    NeuralNetwork(string fileName, string depName, double learningRate, string cost) :
            dataset(Dataset<T>(fileName, depName)), numLayers(0),
            outputLayer(nullptr), inputLayer(nullptr), learningRate(learningRate), dataPartitioned(false), costType(cost) {

        setCostFunction(cost);
    }

    // TODO--test this
    NeuralNetwork(Dataset<T> data, double learningRate, string cost) :
            dataset(data), numLayers(0), outputLayer(nullptr), inputLayer(nullptr), learningRate(learningRate),
            dataPartitioned(false), costType(cost) {

        setCostFunction(cost);

    }

    // For JSON deserialization
    NeuralNetwork(Dataset<T> data, string cost, bool dataPartitioned, Matrix<T> inTrain, Matrix<T> outTrain,
                    Matrix<T> inValidate, Matrix<T> outValidate, double learningRate) :
                        dataset(data), costType(cost), dataPartitioned(dataPartitioned), numLayers(0),
                        outputLayer(nullptr), inputLayer(nullptr), learningRate(learningRate)  {

        setCostFunction(cost);

        // This constructor assumes dataPartitioned = true
        inputTrain = inTrain;
        outputTrain = outTrain;
        inputValidate = inValidate;
        outputValidate = outValidate;

    }

    NeuralNetwork(Dataset<T> data, string cost, bool dataPartitioned, double learningRate) :
                        dataset(data), costType(cost), dataPartitioned(dataPartitioned), numLayers(0),
                        outputLayer(nullptr), inputLayer(nullptr), learningRate(learningRate)  {

        setCostFunction(cost);

    }

    // Copy constructor
    NeuralNetwork(const NeuralNetwork& other) {
        // datasetFilePath = other.getDatasetFilePath();
        dataset = other.getDataset();
        dataPartitioned = other.getDataPartitioned();

        if (dataPartitioned == true) {
            inputTrain = other.getTrainingInput();
            outputTrain = other.getTrainingOutput();
            inputValidate = other.getValidationInput();
            outputValidate = other.getValidationOutput();
        }

        costType = other.getCostType();
        numLayers = other.getNumLayers();
        inputLayer = other.getInputLayer();
        outputLayer = other.getOutputLayer();
        learningRate = other.getLearningRate();

    }

    NeuralNetwork() : dataPartitioned(false), numLayers(0),
                        inputLayer(nullptr), outputLayer(nullptr), learningRate(0.0), costType("") {}

    // NeuralNetwork() : datasetFilePath(""), dataPartitioned(false), numLayers(0),
    //                     inputLayer(nullptr), outputLayer(nullptr), learningRate(0.0), costType("") {}

    ~NeuralNetwork() {
        // delete inputLayer;
        // delete outputLayer;
    }

    NeuralNetwork& operator=(const NeuralNetwork& other) {
        if (this == &other) {
            return *this;
        }

        // datasetFilePath = other.getDatasetFilePath();
        dataset = other.getDataset();
        dataPartitioned = other.getDataPartitioned();

        if (dataPartitioned == true) {
            inputTrain = other.getTrainingInput();
            outputTrain = other.getTrainingOutput();
            inputValidate = other.getValidationInput();
            outputValidate = other.getValidationOutput();
        }

        costType = other.getCostType();
        numLayers = other.getNumLayers();
        inputLayer = other.getInputLayer();
        outputLayer = other.getOutputLayer();
        learningRate = other.getLearningRate();

        return *this;
    }

    bool operator==(const NeuralNetwork& other) const {
        // TODO--refactor so that the dataset filepath isn't necessary to run the network--it could probably be removed entirely
        // if (datasetFilePath != other.getDatasetFilePath()) {
        //     return false;
        // }

        if (dataset != other.getDataset()) {
            return false;
        }

        if (dataPartitioned != other.getDataPartitioned()) {
            return false;
        }

        if (other.getDataPartitioned() == true) {
            if (inputTrain != other.getTrainingInput()) {
                return false;
            }

            if (outputTrain != other.getTrainingOutput()) {
                return false;
            }

            if (inputValidate != other.getValidationInput()) {
                return false;
            }

            if (outputValidate != other.getValidationOutput()) {
                return false;
            }
        }

        if (costType != other.getCostType()) {  // Covers both cost and costDerivative
            return false;
        }

        if (numLayers != other.getNumLayers()) {
            return false;
        }

        // Check that all layers are equal
        if (numLayers != 0) {
            Layer<T> *current = inputLayer;
            Layer<T> *currentOther = other.getInputLayer();

            while (current != nullptr && currentOther != nullptr) {

                if (*current != *currentOther) {
                    return false;
                }

                current = current->getNextLayer();
                currentOther = currentOther->getNextLayer();
            }
        }


        if (learningRate != other.getLearningRate()) {
            return false;
        }

        return true;
    }

    void addLayer(Layer<T> *newLayer) {
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

    void addLayer(int numNodes, string activationType) {

        int prevNodes;
        if (outputLayer == nullptr) {
            prevNodes = dataset.getNumFields();
        } else {
            prevNodes = outputLayer->getNumNodes();
        }

        Layer<T> *newLayer = new Layer<T>(numNodes, prevNodes, activationType);

        addLayer(newLayer);
    }

    // void addLayer(int numNodes, string activationType) {
    //
    //     int prevNodes;
    //     if (outputLayer == nullptr) {
    //         prevNodes = dataset.getNumFields();
    //     } else {
    //         prevNodes = outputLayer->getNumNodes();
    //     }
    //
    //     Layer<T> *newLayer = new Layer<T>(numNodes, prevNodes, activationType);
    //
    //     // Reset layer pointers
    //     if (inputLayer == nullptr) {
    //         inputLayer = newLayer;
    //     }
    //
    //     if (outputLayer == nullptr) {
    //         outputLayer = newLayer;
    //     } else {
    //
    //         outputLayer->setNextLayer(newLayer);
    //         newLayer->setPreviousLayer(outputLayer);
    //         outputLayer = newLayer;
    //     }
    //
    //     numLayers++;
    // }

    void randomizeLayers(T max) {
        randomizeLayers(-max, max);
    }

    void randomizeLayers(T lowBound, T highBound) {
        Layer<T> *current = inputLayer;
        while (current != nullptr) {
            current->randomizeLayer(lowBound, highBound);
            current = current->getNextLayer();
        }
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
        // cout << "TRAINING" << endl;
        cout << "[";
        for (int epoch = 0; epoch < epochs; epoch++) {
            // cout << "Epoch " << epoch + 1 << ":" << endl;

            // Feed forward
            output = feedForwardFull(inputTrain);
            // cost = costFunction(output, outputTrain);
            // cout << cost << endl;


            // Backpropagation
            backpropagateFull(inputTrain, output, outputTrain);

            // Display progress bar
            progressBar(epoch, epochs);
        }
        cout << "]" << endl;

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

    // string getDatasetFilePath() const {
    //     return datasetFilePath;
    // }

    string getCostType() const {
        return costType;
    }

    Dataset<T> getDataset() const {
        return dataset;
    }

    bool getDataPartitioned() const {
        return dataPartitioned;
    }

    Matrix<T> getTrainingInput() const {
        checkDataPartitioned();
        return inputTrain;
    }

    Matrix<T> getTrainingOutput() const {
        checkDataPartitioned();
        return outputTrain;
    }

    Matrix<T> getValidationInput() const {
        checkDataPartitioned();
        return inputValidate;
    }

    Matrix<T> getValidationOutput() const {
        checkDataPartitioned();
        return outputValidate;
    }

    Matrix<T> (*getCostFunction() const)(Matrix<T>, Matrix<T>) {
        return costFunction;
    }

    Matrix<T> (*getCostFunctionDerivative() const)(Matrix<T>, Matrix<T>) {
        return costFunctionDerivative;
    }

    int getNumLayers() const {
        return numLayers;
    }

    Layer<T>* getInputLayer() const {
        return inputLayer;
    }

    Layer<T>* getOutputLayer() const {
        return outputLayer;
    }

    double getLearningRate() const {
        return learningRate;
    }

    Matrix<T> predict(Matrix<T> input) {
        if (input.getNumRows() != dataset.getNumFields()) {
            throw NeuralNetworkException("Input matrix has incorrect dimensions.");
        }

        return feedForwardFull(input);
    }

    void save(string filename) {

    }

private:
    // string datasetFilePath;
    Dataset<T> dataset;

    string costType;

    // Keeps track of whether the dataset has been partitioned or not (if inputTrain, outputTrain, etc. have been declared)
    bool dataPartitioned;

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



    void checkDataPartitioned() const {
        if (!dataPartitioned) {
            throw NeuralNetworkException("Training and validation data are not properly partitioned. Be sure to call 'partitionDataset' before training network.");
        }
    }

    void progressBar(int current, int total) {
        double percentProgress = 1000.0 * current / total;
        // cout << percentProgress << endl;

        if (static_cast<int>(percentProgress) % 10 == 0) {
            cout << "=";
        }
    }

    void setCostFunction(string cost) {
        if (cost == "RMSE") {
            costFunction = rmse;
            costFunctionDerivative = rmseDerivative;
        }
        else if (cost == "MSE") {
            costFunction = mse;
            costFunctionDerivative = mseDerivative;
        }
    }

};


// Functions to serialize and deserialize NeuralNetwork objects as JSON strings
template <typename T>
void to_json(nlohmann::json& j, const NeuralNetwork<T>& network) {
    j["dataset"] = network.getDataset();
    j["costType"] = network.getCostType();
    j["dataPartitioned"] = network.getDataPartitioned();

    if (network.getDataPartitioned()) {
        j["inputTrain"] = network.getTrainingInput();
        j["outputTrain"] = network.getTrainingOutput();
        j["inputValidate"] = network.getValidationInput();
        j["outputValidate"] = network.getValidationOutput();
    }

    j["numLayers"] = network.getNumLayers();

    // Add layers
    Layer<T>* current = network.getInputLayer();
    j["layers"] = nlohmann::json::array();
    while (current != nullptr) {
        j["layers"].push_back(*current);

        current = current->getNextLayer();
    }

    j["learningRate"] = network.getLearningRate();

}

template <typename T>
void from_json(const nlohmann::json& j, NeuralNetwork<T>& network) {
    if (j["dataPartitioned"] == true) {
        network = NeuralNetwork<T>(j.at("dataset"), j["costType"], j.at("dataPartitioned"),
                        j.at("inputTrain"), j.at("outputTrain"), j.at("inputValidate"),
                        j.at("outputValidate"), j.at("learningRate"));
    } else {
        Dataset<T> dataset = j.at("dataset").get<Dataset<T>>(); // Necessary to remove constructor ambiguity
        network = NeuralNetwork<T>(dataset, j["costType"], j.at("dataPartitioned"), j.at("learningRate"));
    }

    // Add layers
    for (int i = 0; i < j.at("numLayers"); i++) {
        Layer<T> *newLayer = new Layer<T>(j.at("layers").at(i).get<Layer<T>>());
        network.addLayer(newLayer);
    }
}




// RMSE
double rmse(Matrix<double> actual, Matrix<double> expected) {
    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmse()");
    }

    return sqrt(pow(actual - expected, 2).sum() / expected.getNumCols()); // RMSE
}

Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected) {

    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmseDerivative(): matrices with dimensions " + actual.dims() + " and " + expected.dims() + " are not compatible.");
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