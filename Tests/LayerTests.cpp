#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <functional>
#include <nlohmann/json.hpp>

#include "Layer.h"

using namespace std;



TEST_CASE("Layer initialization and minimization", "[Layer]") { // TODO

    Matrix<double> input(5, 10); // 5 dependent variables, 10 data points
    input.randomize();

    Layer<double> layer(10, 5, "Sigmoid");

    layer.feedForward(input);

}


TEST_CASE("Layer to_string()", "[Layer]") {
    Layer<double> layer(10, 5, "Leaky ReLU");

    string layerString = to_string(layer);

    string layerStringBeginning = "numNodes: 10\nactivationType: Leaky ReLU\nweights:\n[";

    CHECK(layerString.substr(0, layerStringBeginning.length()) == layerStringBeginning);
}

TEST_CASE("Layer equality", "[Layer]") {
    Layer<double> layer(10, 5, "Sigmoid");
    Layer<double> layer2(10, 5, "Sigmoid");

    CHECK(layer == layer2);
    CHECK_FALSE(layer != layer2);

    Layer<double> layer3(11, 5, "Sigmoid");
    CHECK(layer != layer3);
}

TEST_CASE("Layer copy constructor", "[Layer]") {
    Layer<double> layer(10, 5, "Sigmoid");
    Layer<double> layer2 = layer;

    CHECK(layer == layer2);
}

TEST_CASE("Layer copy assignment", "[Layer]") {
    Layer<double> layer(10, 5, "Sigmoid");
    Layer<double> layer2(11, 6, "ReLU");

    layer = layer2;

    CHECK(layer == layer2);
}

TEST_CASE("Layer JSON serialization", "[Layer]") {
    Layer<double> layer(10, 5, "Sigmoid");

    nlohmann::json layerJSON = layer;
    Layer<double> layerFromJSON = layerJSON.get<Layer<double>>();

    CHECK(layer == layerFromJSON);
}



// TODO--test cases for all the other Layer methods