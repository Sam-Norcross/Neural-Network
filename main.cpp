#include <iostream>
#include "Matrix.h"
#include "Dual.h"
#include "Dataset.h"
#include "NeuralNetwork.h"
using namespace std;

// double relTol(double current, double previous);

int main() {

    cout << "START:" << endl;

    string filepath = "Datasets/BostonHousing.csv";

    NeuralNetwork<double> network(filepath, "medv", 0.05, "MSE");

    network.addLayer(256, "ReLU");
    network.addLayer(128, "ReLU");
    network.addLayer(64, "ReLU");
    network.addLayer(1, "Linear");


    network.partitionDataset(0.75);

    network.trainNetwork(100); // TODO--add simple progress bar?



    return 0;
}



// double relTol(double current, double previous) {
//     return (previous - current) / previous;
// }