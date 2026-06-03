#include <catch2/catch_test_macros.hpp>

#include "Layer.h"

TEST_CASE("Layer initialization and minimization", "[Layer]") {
    Layer<double> layer(2, 2, relu);

}