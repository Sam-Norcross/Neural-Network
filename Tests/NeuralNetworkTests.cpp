#include <catch2/catch_test_macros.hpp>
#include <string>

#include "NeuralNetwork.h"
using namespace std;

TEST_CASE("NeuralNetwork initialization and feed forward", "[NeuralNetwork]") {
    // string filepath = "Datasets/BostonHousing.csv";
    string filepath = "Datasets/auto-mpg.csv";

    NeuralNetwork<double> network(filepath, "medv", 0.01, "RMSE");

    network.addLayer(64, "ReLU");
    network.addLayer(32, "ReLU");
    network.addLayer(1, "Linear");


    network.partitionDataset(0.75);

    // Matrix<double> inputTrain = network.getTrainingInput();

    // cout << network.feedForwardFull(inputTrain).dims();

    // TODO--add test cases for partitionDataset, check that the dimensions of the feedForward output are correct, etc.




    // TODO--probably make this vvv a separate test case
    network.trainNetwork(100);

}

TEST_CASE("Classification architecture", "[NeuralNetwork]") {
    string filepath = "Datasets/HalfMoonDataset.csv";

    NeuralNetwork<double> network(filepath, "moon_id", 0.05, "RMSE");

    network.addLayer(100, "ReLU");
    network.addLayer(50, "Sigmoid");
    network.addLayer(1, "Sigmoid");

    network.partitionDataset(0.75);

    network.trainNetwork(500);  // TODO--costs are lower with values randomized within (-1, 1)
    // TODO--build functionality to export this data for visualization in Julia or Python (or just find a good C++ plotting library)

}