#include <catch2/catch_test_macros.hpp>
#include <iostream>
using namespace std;

#include "Layer.h"

//TODO--Layer shouldn't need explicit references to these libraries
#include "Matrix.h"
#include "Dual.h"

TEST_CASE("Layer initialization and minimization", "[Layer]") {

    Dual<int, int> arr[4] = {Dual<int, int>(0), Dual<int, int>(-1), Dual<int, int>(-2), Dual<int, int>(3)};
    Matrix mat(2, 2, arr);

    // Matrix<double> mat(2, 2);
    // mat.randomize();

    // mat.display();
    // mat = relu(mat);
    //
    // mat.display();
    //
    // mat.get(0, 0).display();    // TODO--this needs to be (0, 0), not (0, 1)

    Layer<double, double> layer(2, 2, relu);

    // mat = sigmoid(mat);
    //
    // Dual<double, double> arr2[4] = {Dual<double, double>(0), Dual<double, double>(-1), Dual<double, double>(-2), Dual<double, double>(3)};
    // Matrix mat2(2, 2, arr2);
    //
    // mat2 = sigmoid(mat2);
    // mat2.display();

}