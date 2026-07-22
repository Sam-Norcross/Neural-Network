#include <catch2/catch_test_macros.hpp>
#include <string>

#include "NeuralNetwork.h"
#include "Matrix.h"
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

TEST_CASE("Regression architecture: boston housing dataset", "[NeuralNetwork]") {
    string filepath = "Datasets/BostonHousing.csv";

    NeuralNetwork<double> network(filepath, "medv", 0.05, "MSE");

    network.addLayer(256, "ReLU");
    network.addLayer(128, "ReLU");
    network.addLayer(64, "ReLU");
    network.addLayer(1, "Linear");


    network.partitionDataset(0.75);

    network.trainNetwork(100); // TODO--add simple progress bar?

}

TEST_CASE("Classification architecture", "[NeuralNetwork]") {
    string filepath = "Datasets/HalfMoonDataset.csv";

    NeuralNetwork<double> network(filepath, "moon_id", 0.15, "RMSE");
    
    network.addLayer(100, "ReLU");
    network.addLayer(50, "Sigmoid");
    network.addLayer(1, "Sigmoid");

    network.randomizeLayers(1.0);

    network.partitionDataset(0.75);

    network.trainNetwork(1000);
    // TODO--build functionality to export this data for visualization in Julia or Python (or just find a good C++ plotting library)


    Matrix<double> predictedOutput = network.predict(network.getValidationInput());

    cout << "Validation data input:\n";
    network.getValidationInput().display();

    cout << "Validation data output:\n";
    predictedOutput.display();



    // Run the network for a grid of values to test decision boundary

    // Create grid
    double xMin = -2.5;
    double xMax = 2.5;
    double yMin = -1.5;
    double yMax = 1.5;
    double h = 0.1;

    int numXPts = (xMax - xMin) / h + 1;
    int numYPts = (yMax - yMin) / h + 1;

    Matrix<double> gridInput(2, numXPts * numYPts);

    int i = 0;
    for (int yInc = 0; yInc < numYPts; yInc++) {
        for (int xInc = 0; xInc < numXPts; xInc++) {

            gridInput.get(0, i) = xMin + xInc * h;
            gridInput.get(1, i) = yMin + yInc * h;
            i++;
        }
    }

    cout << "Grid input:\n";
    gridInput.display();

    // Run network
    Matrix<double> gridPredictions = network.predict(gridInput);

    cout << "Grid output:\n";
    gridPredictions.display();
}