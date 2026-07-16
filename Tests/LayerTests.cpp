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

    Layer<double> layer(10, 5, sigmoid, sigmoidDerivative);
    // Layer<double> layer(5, 5, sigmoidTEST<double>, sigmoidDerivative);

    layer.feedForward(input);

}
