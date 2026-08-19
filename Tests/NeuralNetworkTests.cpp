#include <catch2/catch_test_macros.hpp>
#include <string>

#include "NeuralNetwork.h"
#include "Matrix.h"
using namespace std;

TEST_CASE("NeuralNetwork equality", "[NeuralNetwork]") {
    string filepath = "Datasets/BostonHousing.csv";
    NeuralNetwork<double> network(filepath, "medv", 0.01, "RMSE");
    network.addLayer(64, "ReLU");
    network.addLayer(32, "ReLU");
    network.addLayer(1, "Linear");

    NeuralNetwork<double> network2(filepath, "medv", 0.01, "RMSE");
    network2.addLayer(64, "ReLU");
    network2.addLayer(32, "ReLU");
    network2.addLayer(1, "Linear");

    CHECK(network == network2);
}

TEST_CASE("NeuralNetwork copy constructor", "[NeuralNetwork]") {
    string filepath = "Datasets/BostonHousing.csv";
    NeuralNetwork<double> network(filepath, "medv", 0.01, "RMSE");
    network.addLayer(64, "ReLU");
    network.addLayer(32, "ReLU");
    network.addLayer(1, "Linear");

    NeuralNetwork<double> network2 = network;

    CHECK(network == network2);
}

TEST_CASE("NeuralNetwork copy assignment operator", "[NeuralNetwork]") {
    string filepath = "Datasets/BostonHousing.csv";
    NeuralNetwork<double> network(filepath, "medv", 0.01, "RMSE");
    network.addLayer(64, "ReLU");
    network.addLayer(32, "ReLU");
    network.addLayer(1, "Linear");

    NeuralNetwork<double> network2(filepath, "medv", 0.5, "MSE");
    network2 = network;

    CHECK(network == network2);
}

TEST_CASE("NeuralNetwork constructor with Dataset", "[NeuralNetwork]") {
    int numPts = 1000;
    Matrix xVals = linspace<double>(0, 10, numPts); // TODO--if there are too many data points, the network cost goes to infinity
    Matrix yVals = 2 * xVals + 3;

    Dataset dataset(xVals, yVals);

    NeuralNetwork network(dataset, 0.01, "MSE");

    // Copied from Linear Regression test
    network.addLayer(1, "Linear");
    network.partitionDataset(numPts - 1);

    network.trainNetwork(4000);

    double tol = 1e-16;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

TEST_CASE("NeuralNetwork initialization and feed forward", "[NeuralNetwork]") {
    string filepath = "Datasets/BostonHousing.csv";
    // string filepath = "Datasets/auto-mpg.csv";   // TODO--no error is thrown when this dataset is used with depName = "medv" (should be "mpg")

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

TEST_CASE("NeuralNetwork JSON serialization", "[Layer]") {
    string filepath = "Datasets/BostonHousing.csv";
    NeuralNetwork<double> network(filepath, "medv", 0.01, "RMSE");
    network.addLayer(64, "ReLU");
    network.addLayer(32, "ReLU");
    network.addLayer(1, "Linear");

    nlohmann::json networkJSON = network;
    NeuralNetwork<double> networkFromJSON = networkJSON.get<NeuralNetwork<double>>();

    CHECK(network == networkFromJSON);
}

TEST_CASE("NeuralNetwork JSON saving and loading", "[NeuralNetwork]") { // Overfitting should result in near-zero loss
    string filepath = "Tests/TestDatasets/TestData1.csv";

    NeuralNetwork<double> network(filepath, "Threes", 0.01, "MSE");

    network.addLayer(128, "Leaky ReLU");
    network.addLayer(64, "Sigmoid");
    network.addLayer(32, "ReLU");
    network.addLayer(1, "Linear");

    network.partitionDataset(2);

    network.trainNetwork(50);

    double tol = 1e-16;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);

    // Save network
    string nnFilename = "Tests/SavedNeuralNetwork.json";
    network.save(nnFilename);

    // Load saved network   //TODO--maybe add a function in NeuralNetwork.h (outside the class) to do this more cleanly?
    NeuralNetwork<double> loadedNetwork = NeuralNetwork<double>();
    loadedNetwork.load(nnFilename);

    CHECK(loadedNetwork == network);

    CHECK(loadedNetwork.trainingCost() < tol);
    CHECK(loadedNetwork.validationCost() < tol);

    CHECK(loadedNetwork.trainingCost() == network.trainingCost());
    CHECK(loadedNetwork.validationCost() == network.validationCost());
}

// Known neural network test cases
TEST_CASE("NeuralNetwork overfitting", "[NeuralNetwork]") { // Overfitting should result in near-zero loss
    string filepath = "Tests/TestDatasets/TestData1.csv";

    NeuralNetwork<double> network(filepath, "Threes", 0.01, "MSE");

    network.addLayer(128, "Leaky ReLU");
    network.addLayer(64, "Sigmoid");
    network.addLayer(32, "ReLU");
    network.addLayer(1, "Linear");

    network.partitionDataset(2);

    network.trainNetwork(20);

    double tol = 1e-16;
    CHECK(network.trainingCost() < tol);
}

TEST_CASE("Identity with MSE", "[NeuralNetwork]") {
    string filepath = "Tests/TestDatasets/Identity.csv";    // Contains x and y values for y = x

    NeuralNetwork<double> network(filepath, "y", 0.1, "MSE");
    network.addLayer(1, "Linear");
    network.partitionDataset(9);

    network.trainNetwork(300);

    double tol = 1e-16;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);

    Matrix predictInput = linspace<double>(3.5, 8.5, 5).transpose();    // Transposed to a row vector
    Matrix predictionDiff = abs(network.predict(predictInput) - predictInput);

    double diffTol = 1e-13;
    CHECK(predictionDiff.get(0, 0) < diffTol);
    CHECK(predictionDiff.get(0, 1) < diffTol);
    CHECK(predictionDiff.get(0, 2) < diffTol);
    CHECK(predictionDiff.get(0, 3) < diffTol);
    CHECK(predictionDiff.get(0, 4) < diffTol);
}

// Known neural network test cases
TEST_CASE("Identity with RMSE", "[NeuralNetwork]") {
    string filepath = "Tests/TestDatasets/Identity.csv";    // Contains x and y values for y = x

    NeuralNetwork<double> network(filepath, "y", 0.0001, "RMSE");
    network.addLayer(1, "Linear");
    network.partitionDataset(9);

    network.trainNetwork(100000);

    double tol = 0.005;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

// TODO--tweak these vvv tests to increase efficiency

TEST_CASE("Constant function", "[NeuralNetwork]") {
    string filepath = "Tests/TestDatasets/Constant.csv";    // Contains x and y values for y = 5

    NeuralNetwork<double> network(filepath, "y", 0.1, "MSE");
    network.addLayer(1, "Linear");
    network.partitionDataset(9);

    network.trainNetwork(200);

    double tol = 1e-16;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

TEST_CASE("Linear regression", "[NeuralNetwork]") {
    string filepath = "Tests/TestDatasets/LinearRegression.csv";    // Contains x and y values for y = 3x + 2

    NeuralNetwork<double> network(filepath, "y", 0.1, "MSE");
    network.addLayer(1, "Linear");
    network.partitionDataset(9);

    network.trainNetwork(200);

    double tol = 1e-16;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

// Two-layer linear regression
TEST_CASE("Two-layer linear regression", "[NeuralNetwork]") {
    string filepath = "Tests/TestDatasets/LinearRegression2Layer.csv";
    // Contains x1, x2, and y values for y = 3 * x1 - 4 * x2 + 5

    NeuralNetwork<double> network(filepath, "y", 0.01, "MSE");
    network.addLayer(1, "Linear");
    network.addLayer(1, "Linear");
    network.partitionDataset(19);

    network.trainNetwork(50);

    double tol = 1e-16;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

TEST_CASE("Three-layer network (XOR)", "[NeuralNetwork]") {
    string filepath = "Tests/TestDatasets/XOR.csv";
    // Contains x1, x2, and y values for y = XOR(x1, x2)

    NeuralNetwork<double> network(filepath, "y", 0.05, "BCE");
    network.addLayer(10, "ReLU");
    network.addLayer(10, "ReLU");
    network.addLayer(1, "Sigmoid");
    network.partitionDataset(38);

    network.trainNetwork(5000);

    double tol = 0.005;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

// TODO--implement batch training
// TODO--add test case to read in pre-trained network from JSON



// More complex datasets
TEST_CASE("Regression architecture: boston housing dataset", "[NeuralNetwork]") {
    string filepath = "Datasets/BostonHousing.csv";

    NeuralNetwork<double> network(filepath, "medv", 0.001, "MSE");

    network.addLayer(30, "ReLU");
    network.addLayer(15, "ReLU");
    network.addLayer(8, "ReLU");
    network.addLayer(5, "ReLU");
    network.addLayer(1, "Linear");

    network.partitionDataset(0.85);    // TODO--add a test case for partitionDataset()

    network.trainNetwork(1000);

    network.getTrainingOutput().display();
    network.predict(network.getTrainingInput()).display();


    double tol = 1;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

TEST_CASE("Regression architecture: mileage dataset", "[NeuralNetwork]") {
    string filepath = "Datasets/auto-mpg.csv";

    NeuralNetwork<double> network(filepath, "mpg", 0.01, "MSE");
    // TODO--no error is thrown if an incorrect dependent name is entered

    // network.addLayer(30, "ReLU");
    network.addLayer(15, "ReLU");
    network.addLayer(8, "ReLU");
    network.addLayer(5, "ReLU");
    network.addLayer(1, "Linear");

    network.partitionDataset(0.85);

    network.trainNetwork(10000);

    network.getTrainingOutput().display();
    network.predict(network.getTrainingInput()).display();


    double tol = 1;
    CHECK(network.trainingCost() < tol);
    CHECK(network.validationCost() < tol);
}

TEST_CASE("Classification architecture", "[NeuralNetwork]") {
    // See the Julia notebooks for dataset generation and visualization
    string filepath = "Datasets/HalfMoonDataset.csv";

    NeuralNetwork<double> network(filepath, "moon_id", 0.5, "BCE");
    
    network.addLayer(10, "ReLU");
    network.addLayer(5, "Sigmoid");
    network.addLayer(1, "Sigmoid");

    network.randomizeLayers(1.0);

    network.partitionDataset(0.75);

    network.trainNetwork(1000);
    // TODO--build functionality to export this data for visualization in Julia or Python (or just find a good C++ plotting library)
    // Maybe add a function to save the output in a csv?


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