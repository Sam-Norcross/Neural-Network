#include <catch2/catch_test_macros.hpp>
#include "Matrix.h"
#include "Dual.h"
#include <cmath>

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

TEST_CASE("Autodiff power", "[Dual]") {
    Dual num = Dual(5.0, 1.0);

    Dual result = pow(num, 2.0);
    CHECK(result.getValue() == 25.0);
    CHECK(result.getDerivative() == 10.0);
}

TEST_CASE("Autodiff exponentials", "[Dual]") {
    // f(x) = 2^x, f'(x) = 2^x * ln(2)
    // f(3) = 8, f'(3) = 8 * ln(2)
    Dual num = Dual(3.0, 1.0);

    Dual result = pow(2.0, num);
    CHECK(result.getValue() == 8.0);
    CHECK(result.getDerivative() == 8.0 * log(2.0));
}

TEST_CASE("Autodiff exp()", "[Dual]") {
    // f(x) = 5 exp(-(2x)^2) + 2, f'(x) = -5 exp(-(2x)^2) * 8x
    // f(-3) = 5 exp(-36) + 2, f'(-3) = 120 exp(-36)

    Dual num = Dual(-3.0, 1.0);

    Dual result = 5.0 * exp(-1.0 * pow(2.0 * num, 2.0)) + 2;

    CHECK(result.getValue() == 5.0 * exp(-36.0) + 2.0);
    CHECK(result.getDerivative() == 120.0 * exp(-36.0));
}

TEST_CASE("Autodiff logarithms", "[Dual]") {
    // f(x) = log(2x + 1) + log10(5x) + log2(x / 2), f'(x) = 2 / (2x + 1) + log(10) / x + log(2) / x
    // f(2) = log(5) + log10(10) + log2(2 / 2), f'(2) = 2 / 5 + 1 / (2 * log(10)_ + 1 / (2 * log(2))

    Dual num = Dual(2.0, 1.0);

    Dual result = log(2.0 * num + 1.0) + log10(5.0 * num) + log2(num / 2.0);
    CHECK(result.getValue() == log(5.0) + 1);
    CHECK(result.getDerivative() == 2.0 / 5.0 + 1.0 / (2.0 * log(10.0)) + 1 / (2.0 * log(2.0)));
}

TEST_CASE("Autodiff with matrices", "[Dual]") {
    // f(x) = 3x^2 - 2x + 1, f'(x) = 6x - 2
    // x = [1 2; 3 4]
    // f(x) = [2 9; 22 41]
    // f'(x) = [4 10; 16 22]

    Dual<double> arr[4] = {Dual(1.0, 1.0), Dual(2.0, 1.0), Dual(3.0, 1.0), Dual(4.0, 1.0)};
    Matrix mat(2, 2, arr);

    mat.display();

    Matrix fMat = 3.0 * pow(mat, 2.0);// - 2.0 * mat + 1.0;

    fMat.display();

    // TODO--fix this too!
    // Matrix<Dual<double>> mat(2, 2);// = Matrix<Dual>(2, 2);
}