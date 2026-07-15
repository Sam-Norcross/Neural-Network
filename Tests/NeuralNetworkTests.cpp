#include <catch2/catch_test_macros.hpp>
#include <string>

#include "NeuralNetwork.h"
using namespace std;

#include "Matrix.h" // TODO--remove eventually if possible

TEST_CASE("NeuralNetwork initialization and feed forward", "[NeuralNetwork]") {
    string filepath = "Datasets/BostonHousing.csv";

    NeuralNetwork<double> network(filepath, "medv", 0.01, "RMSE");

    network.addLayer(10, "ReLU");
    network.addLayer(20, "Sigmoid");
    network.addLayer(1, "ReLU");


    network.partitionDataset(0.75);

    Matrix<double> inputTrain = network.getTrainingInput();

    cout << network.feedForwardFull(inputTrain).dims();

    // TODO--add test cases for partitionDataset, check that the dimensions of the feedForward output are correct, etc.




    // TODO--probably make this vvv a separate test case
    network.trainNetwork(10);

}