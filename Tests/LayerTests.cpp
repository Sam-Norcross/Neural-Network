#include <catch2/catch_test_macros.hpp>

#include "Layer.h"

//TODO--Layer shouldn't need explicit references to these libraries
#include "Matrix.h"
#include "Dual.h"

TEST_CASE("Layer initialization and minimization", "[Layer]") {

    Matrix<Dual<int, int>> mat(2, 2);
    // Matrix<double> mat(2, 2);
    mat.randomize();

    mat.display();
    mat = sigmoid(mat);

    mat.display();


    // Layer<double> layer(2, 2, relu);

}