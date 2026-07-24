#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <functional>

#include "Layer.h"

using namespace std;

//TODO--Layer shouldn't need explicit references to these libraries
// #include "Matrix.h"


TEST_CASE("Layer initialization and minimization", "[Layer]") {

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