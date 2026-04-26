#include <catch2/catch_test_macros.hpp>
#include "Matrix.h"
#include "Dual.h"

TEST_CASE("Autodiff scalar multiplication and addition", "[Dual]") {
    // f(x) = 2x + 3, f'(x) = 2
    // f(3) = 9, f'(3) = 2
    Dual num = Dual(3.0, 1.0);

    Dual result = 2.0 * num + 3.0;
    CHECK(result.getValue() == 9.0);
    CHECK(result.getDerivative() == 2.0);
}

TEST_CASE("Autodiff scalar addition/subtraction/multiplication/division", "[Dual]") {
    // f(x) = 5x + 3 - x/2 - 1, f'(x) = 5 - 1/2 = 4.5
    // f(2) = 11, f'(1) = 5.5
    Dual num = Dual(2.0, 1.0);

    Dual result = 5.0 * num + 3.0 - num / 2.0 - 1.0;
    CHECK(result.getValue() == 11);
    CHECK(result.getDerivative() == 4.5);
}

TEST_CASE("Autodiff exponent", "[Dual]") {
    Dual num = Dual(5.0, 1.0);

    Dual result = pow(num, 2.0);
    CHECK(result.getValue() == 25.0);
    CHECK(result.getDerivative() == 10.0);
}