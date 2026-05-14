#include <catch2/catch_test_macros.hpp>
#include "Matrix.h"
#include "Dual.h"
#include <cmath>

TEST_CASE("Dual cast to double", "[Dual]") {
    double num = 5.0;
    Dual result = Dual(num, 1.0);

    Dual<double, double> result2 = num;

    CHECK(result2 == result);
}

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

TEST_CASE("Autodiff scalar division", "[Dual]") {
    // f(x) = 1/x, f'(x) = -1/x^2
    // f(5) = 1/5, f'(5) = -1/25
    Dual num = Dual(5.0, 1.0);

    Dual result = 1.0 / num;

    CHECK(result.getValue() == 1.0 / 5.0);
    CHECK(result.getDerivative() == -1.0 / 25.0);
}

TEST_CASE("Autodiff Dual division", "[Dual]") {
    Dual num1 = Dual(4.0, 1.0);
    Dual num2 = Dual(2.0, 1.0);

    Dual result = num1 / num2;

    CHECK(result.getValue() == 2.0);
    // CHECK(result.getDerivative() == /*TODO*/); TODO--what does this actually mean? What should the result be?

    Dual num3 = Dual(2.0, 0.0);
    Dual num4 = Dual(4.0, 1.0);

    Dual result2 = num3 / num4;
    Dual result3 = num4 / num3;

    CHECK(result2 == Dual(1.0 / 2.0, -2.0 / pow(4.0, 2.0)));
    CHECK(result3 == Dual(2.0, 1.0 / 2.0));

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

    Dual<double, double> arr[4] = {Dual(1.0, 1.0), Dual(2.0, 1.0), Dual(3.0, 1.0), Dual(4.0, 1.0)};
    Matrix mat(2, 2, arr);

    Matrix fMat = 3.0 * pow(mat, 2.0) - 2.0 * mat + 1.0;

    double vals[4] = {2.0, 9.0, 22.0, 41.0};
    double ders[4] = {4.0, 10.0, 16.0, 22.0};
    Matrix valMat = Matrix(2, 2, vals);
    Matrix derMat = Matrix(2, 2, ders);

    CHECK(getValue(fMat) == valMat);
    CHECK(getDerivative(fMat) == derMat);
}

TEST_CASE("Scalar gradient", "[Dual]") {
    // f(x, y, z) = 2 x^2 y + xyz + 3z
    // Df(x, y, z) = (4x y + yz, 2x^2 + xz, xy + 3)
    // f(1, 2, 3) = 4 + 6 + 9 = 19
    // Df(1, 2, 3) = (8 + 6, 2 + 3, 2 + 3) = (14, 5, 5)

    double arrX[3] = {1.0, 0.0, 0.0};   //TODO--maybe clean up this structure better? Add another Dual constructor that takes in an initializer list or array to remove the extra lines?
    Matrix<double> matX = Matrix<double>(1, 3, arrX);

    double arrY[3] = {0.0, 1.0, 0.0};
    Matrix<double> matY = Matrix<double>(1, 3, arrY);

    double arrZ[3] = {0.0, 0.0, 1.0};
    Matrix<double> matZ = Matrix<double>(1, 3, arrZ);

    Dual x = Dual(1.0, matX);// 1.0);//{1.0, 0.0, 0.0});
    Dual y = Dual(2.0, matY);//{0.0, 1.0, 0.0});
    Dual z = Dual(3.0, matZ);//{0.0, 0.0, 1.0});

    Dual f = 2.0 * pow(x, 2.0) * y + x * y * z + 3.0 * z;

    CHECK(f.getValue() == 19.0);
    CHECK(f.getDerivative().get(0, 0) == 14.0);
    CHECK(f.getDerivative().get(0, 1) == 5.0);
    CHECK(f.getDerivative().get(0, 2) == 5.0);
}