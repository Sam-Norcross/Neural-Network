#include <catch2/catch_test_macros.hpp>
#include "Matrix.h"
#include "Dual.h"

TEST_CASE("Matrix initialization and randomization", "[Matrix]") {
    Matrix<int> mat1(2, 2);

    mat1.randomize();   // Randomizes values within (-10, 10)

    // Check that matrix values are randomized within the bounds
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            CHECK(mat1.get(r, c) > -10);
            CHECK(mat1.get(r, c) < 10);
        }
    }

    // Reset matrix values
    mat1.get(0, 0) = 100;
    mat1.get(0, 1) = 100;
    mat1.get(1, 0) = 100;
    mat1.get(1, 1) = 100;

    // Randomize to within (-5, 5) and recheck
    mat1.randomize(-5, 5);

    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            CHECK(mat1.get(r, c) > -5);
            CHECK(mat1.get(r, c) < 5);
        }
    }

    // Check for a matrix of double values
    Matrix<double> mat2(2, 2);

    mat2.randomize();   // Randomizes values within (-10, 10)

    // Check that matrix values are randomized within the bounds
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 2; c++) {
            CHECK(mat2.get(r, c) > -10);
            CHECK(mat2.get(r, c) < 10);
        }
    }
}

TEST_CASE("Matrix get()", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int arr2[5] = {1, 2, 3, 4, 5};
    Matrix mat2(1, 5, arr2);

    int arr3[3] = {2, 3, 4};
    Matrix mat3(3, 1, arr3);

    int arr4[6] = {1, 2, 3, 4, 5, 6};
    Matrix mat4(3, 2, arr4);

    CHECK(mat1.get(0, 0) == 1);
    CHECK(mat1.get(0, 1) == 2);
    CHECK(mat1.get(1, 0) == 3);
    CHECK(mat1.get(1, 1) == 4);

    CHECK(mat2.get(0, 0) == 1);
    CHECK(mat2.get(0, 4) == 5);

    CHECK(mat3.get(0, 0) == 2);
    CHECK(mat3.get(1, 0) == 3);
    CHECK(mat3.get(2, 0) == 4);

    CHECK(mat4.get(0, 0) == 1);
    CHECK(mat4.get(1, 1) == 4);
    CHECK(mat4.get(2, 1) == 6);
}

TEST_CASE("Matrix getRow() and getCol()", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int row0Arr[2] = {1, 2};
    Matrix row0(1, 2, row0Arr);

    int row1Arr[2] = {3, 4};
    Matrix row1(1, 2, row1Arr);

    int col0Arr[2] = {1, 3};
    Matrix col0(2, 1, col0Arr);

    int col1Arr[2] = {2, 4};
    Matrix col1(2, 1, col1Arr);

    CHECK(mat1.getRow(0) == row0);
    CHECK(mat1.getRow(1) == row1);
    CHECK(mat1.getCol(0) == col0);
    CHECK(mat1.getCol(1) == col1);
}

TEST_CASE("Matrix toString()", "{Matrix}") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);
    string mat1String = "[1, 2\n3, 4]";

    int arr2[5] = {1, 2, 3, 4, 5};
    Matrix mat2(1, 5, arr2);
    string mat2String = "[1, 2, 3, 4, 5]";

    int arr3[3] = {2, 3, 4};
    Matrix mat3(3, 1, arr3);
    string mat3String = "[2\n3\n4]";

    int arr4[6] = {1, 2, 3, 4, 5, 6};
    Matrix mat4(3, 2, arr4);
    string mat4String = "[1, 2\n3, 4\n5, 6]";

    int arr5[6] = {2, 3, 4, 5, 6, 7};
    Matrix mat5(2, 3, arr5);
    string mat5String = "[2, 3, 4\n5, 6, 7]";

    CHECK(mat1.toString() == mat1String);
    CHECK(mat2.toString() == mat2String);
    CHECK(mat3.toString() == mat3String);
    CHECK(mat4.toString() == mat4String);
    CHECK(mat5.toString() == mat5String);
}

TEST_CASE("Matrix Equality", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);
    Matrix mat2(2, 2, arr1);
    CHECK(mat1 == mat2);

    int arr3[4] = {2, 2, 3, 4};
    Matrix mat3(2, 2, arr3);
    CHECK(mat1 != mat3);
}

TEST_CASE("Matrix comparison operators", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int arr2[4] = {2, 3, 4, 5};
    Matrix mat2(2, 2, arr2);

    CHECK(mat1 < mat2);
    CHECK(mat1 <= mat2);
    CHECK(mat2 > mat1);
    CHECK(mat2 >= mat1);

    CHECK(mat1 <= mat1);
    CHECK(mat2 >= mat2);

    CHECK(mat1 > 0);
    CHECK(mat1 >= 1);
    CHECK(mat1 < 5);
    CHECK(mat1 <= 4);
}

TEST_CASE("Matrix copy()", "[Matrix]") {

    // Check assigning a new matrix (mat2) to an existing matrix (mat1)
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    Matrix mat2 = mat1;//.copy();

    CHECK(mat1 == mat2);

    // Check assigning an existing matrix (mat3) to an existing matrix (mat1)
    Matrix<int> mat3(2, 2);
    mat3 = mat1;

    CHECK(mat1 == mat3);
    CHECK(mat2 == mat3);

    // Check that editing the copy doesn't edit the original
    mat3.get(0, 0) = 100;

    CHECK(mat1.get(0, 0) == 1);
    CHECK(mat2.get(0, 0) == 1);
    CHECK(mat3.get(0, 0) == 100);
}

TEST_CASE("Matrix Addition", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);
    Matrix mat2(2, 2, arr1);

    int arr3[4] = {2, 4, 6, 8};
    Matrix mat3(2, 2, arr3);
    CHECK(mat1 + mat2 == mat3);
}

TEST_CASE("Matrix Subtraction", "[Matrix]") {
    int arr1[6] = {1, 2, 3, 4, 5, 6};
    Matrix mat1(2, 3, arr1);
    Matrix mat2(2, 3, arr1);

    int arr3[6] = {0, 0, 0, 0, 0, 0};
    Matrix mat3(2, 3, arr3);
    CHECK(mat1 - mat2 == mat3);
}

TEST_CASE("Matrix Element-Wise Multiplication", "[Matrix]") {
    int arr1[6] = {1, 2, 3, 4, 5, 6};
    Matrix mat1(2, 3, arr1);
    Matrix mat2(2, 3, arr1);

    int arr3[6] = {1, 4, 9, 16, 25, 36};
    Matrix mat3(2, 3, arr3);
    CHECK(mat1 * mat2 == mat3);
}

TEST_CASE("Matrix Element-Wise Division", "[Matrix]") {
    int arr1[6] = {1, 2, 3, 4, 5, 6};
    Matrix mat1(2, 3, arr1);
    Matrix mat2(2, 3, arr1);

    int arr3[6] = {1, 1, 1, 1, 1, 1};
    Matrix mat3(2, 3, arr3);
    CHECK(mat1 / mat2 == mat3);
}

TEST_CASE("Matrix Multiplication 1x1", "[Matrix]") {
    int arr1[1] = {2};
    Matrix mat1(1, 1, arr1);

    int arr2[1] = {3};
    Matrix mat2(1, 1, arr2);

    int arr3[1] = {6};
    Matrix resultMat(1, 1, arr3);

    CHECK(mat1.matMul(mat2) == resultMat);
}

TEST_CASE("Matrix Multiplication 1x2 * 2x1", "[Matrix]") {
    int arr1[2] = {2, 3};
    Matrix mat1(1, 2, arr1);

    int arr2[2] = {3, 4};
    Matrix mat2(2, 1, arr2);

    int arr3[1] = {18};
    Matrix resultMat(1, 1, arr3);

    CHECK(mat1.matMul(mat2) == resultMat);
}

TEST_CASE("Matrix Multiplication 2x1 * 1x2", "[Matrix]") {
    int arr1[2] = {2, 3};
    Matrix mat1(2, 1, arr1);

    int arr2[2] = {3, 4};
    Matrix mat2(1, 2, arr2);

    int arr3[4] = {6, 8, 9, 12};
    Matrix resultMat(2, 2, arr3);

    CHECK(mat1.matMul(mat2) == resultMat);
}

TEST_CASE("Matrix Multiplication 2x2", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int arr2[4] = {2, 3, 4, 5};
    Matrix mat2(2, 2, arr2);

    int arr3[4] = {10, 13, 22, 29};
    Matrix resultMat(2, 2, arr3);

    CHECK(mat1.matMul(mat2) == resultMat);
}

TEST_CASE("Scalar Addition", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int addVal = 5;

    int arr3[4] = {6, 7, 8, 9};
    Matrix mat3(2, 2, arr3);

    CHECK(mat1 + addVal == mat3);
    CHECK(addVal + mat1 == mat3);
}

TEST_CASE("Scalar Subtraction", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int subVal = 5;

    int arr3[4] = {-4, -3, -2, -1};
    Matrix mat3(2, 2, arr3);

    CHECK(mat1 - subVal == mat3);

    int arr4[4] = {4, 3, 2, 1};
    Matrix mat4(2, 2, arr4);

    CHECK(subVal - mat1 == mat4);
}

TEST_CASE("Scalar Multiplication", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int multVal = 5;

    int arr3[4] = {5, 10, 15, 20};
    Matrix mat3(2, 2, arr3);

    CHECK(mat1 * multVal == mat3);
    CHECK(multVal * mat1 == mat3);
}

TEST_CASE("Scalar Division", "[Matrix]") {
    int arr1[4] = {2, 4, 6, 8};
    Matrix mat1(2, 2, arr1);

    int divVal = 2;

    int arr3[4] = {1, 2, 3, 4};
    Matrix mat3(2, 2, arr3);

    CHECK(mat1 / divVal == mat3);
}

TEST_CASE("Matrix transpose() 1x1", "[Matrix]") {
    int arr1[1] = {1};
    Matrix mat1(1, 1, arr1);

    CHECK(mat1.transpose() == mat1);
}

TEST_CASE("Matrix transpose() 2x1", "[Matrix]") {
    int arr1[2] = {1, 2};
    Matrix mat1(2, 1, arr1);

    int arr2[2] = {1, 2};
    Matrix mat2(1, 2, arr2);

    CHECK(mat1.transpose() == mat2);
    CHECK(mat2.transpose() == mat1);
}

TEST_CASE("Matrix transpose() 2x2", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int arr2[4] = {1, 3, 2, 4};
    Matrix mat2(2, 2, arr2);

    CHECK(mat1.transpose() == mat2);
    CHECK(mat2.transpose() == mat1);
}

TEST_CASE("Matrix transpose() 2x3", "[Matrix]") {
    int arr1[6] = {1, 2, 3, 4, 5, 6};
    Matrix mat1(2, 3, arr1);

    int arr2[6] = {1, 4, 2, 5, 3, 6};
    Matrix mat2(3, 2, arr2);

    CHECK(mat1.transpose() == mat2);
    CHECK(mat2.transpose() == mat1);
}

TEST_CASE("Matrix addition with Dual elements", "[Matrix]") {
    Dual<double, double> arr[4] = {Dual(1.0, 1.0), Dual(2.0, 1.0), Dual(3.0, 1.0), Dual(4.0, 1.0)};
    Matrix mat1(2, 2, arr);
    Matrix mat2(2, 2, arr);

    Dual<double, double> resultArr[4] = {Dual(2.0, 2.0), Dual(4.0, 2.0), Dual(6.0, 2.0), Dual(8.0, 2.0)};
    Matrix mat3(2, 2, resultArr);

    CHECK(mat1 + mat2 == mat3);
}

TEST_CASE("Matrix getValue() and getDerivative()", "[Matrix]") {
    Dual<double, double> arr[4] = {Dual(1.0, 1.0), Dual(2.0, 0.0), Dual(3.0, -1.0), Dual(4.0, -2.0)};
    Matrix mat(2, 2, arr);

    double vals[4] = {1.0, 2.0, 3.0, 4.0};
    double ders[4] = {1.0, 0.0, -1.0, -2.0};

    Matrix valMat = Matrix(2, 2, vals);
    Matrix derMat = Matrix(2, 2, ders);

    CHECK(getValue(mat) == valMat);
    CHECK(getDerivative(mat) == derMat);
}

TEST_CASE("Matrix map()", "[Matrix]") {
    int arr[4] = {1, 2, 3, 4};
    Matrix mat(2, 2, arr);

    int arrResult[4] = {2, 3, 4, 5};
    Matrix result(2, 2, arrResult);

    function func = [](int val){return val + 1;};
    mat.map(func);

    CHECK(mat == result);
}

TEST_CASE("Matrix exp()", "[Matrix]") {
    double arr[4] = {1.0, 2.0, 3.0, 4.0};
    Matrix mat(2, 2, arr);

    double arrResult[4] = {exp(1.0), exp(2.0), exp(3.0), exp(4.0)};
    Matrix result(2, 2, arrResult);

    CHECK(exp(mat) == result);
}

TEST_CASE("Scalar / matrix ", "[Matrix]") {
    double arr[4] = {1.0, 2.0, 3.0, 4.0};
    Matrix mat(2, 2, arr);

    double arrResult[4] = {5.0 / 1.0, 5.0 / 2.0, 5.0 / 3.0, 5.0 / 4.0};
    Matrix result(2, 2, arrResult);

    CHECK(5.0 / mat == result);
}

TEST_CASE("Matrix colAdd()", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int arr2[2] = {5, 1};
    Matrix colVec(2, 1, arr2);

    int resArr[4] = {6, 7, 4, 5};
    Matrix result(2, 2, resArr);

    CHECK(mat1.colAdd(colVec) == result);
}

TEST_CASE("Matrix colAdd() 5x1 + 5x1", "[Matrix]") {
    int arr1[5] = {1, 2, 3, 4, 5};
    Matrix mat1(5, 1, arr1);

    int arr2[5] = {3, 6, 9, 12, 15};
    Matrix colVec(5, 1, arr2);

    int resArr[5] = {4, 8, 12, 16, 20};
    Matrix result(5, 1, resArr);

    CHECK(mat1.colAdd(colVec) == result);
}

TEST_CASE("Matrix sumToColVec()", "[Matrix]") {
    int arr1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    Matrix mat1(3, 3, arr1);

    int arr2[3] = {6, 15, 24};
    Matrix result(3, 1, arr2);

    CHECK(mat1.sumToColVec() == result);
}

TEST_CASE("Matrix/matrix +=, -=, *=, and /=", "[Matrix]") {
    int arrMat[4] = {1, 2, 3, 4};
    Matrix originalMat(2, 2, arrMat);
    Matrix mat1(2, 2, arrMat);

    int arrFac[4] = {1, 3, 5, 7};
    Matrix fac(2, 2, arrFac);

    mat1 += fac;
    CHECK(mat1 == originalMat + fac);

    mat1 -= fac;
    CHECK(mat1 == originalMat);

    mat1 *= fac;
    CHECK(mat1 == originalMat * fac);

    mat1 /= fac;
    CHECK(mat1 == originalMat);
}

TEST_CASE("Matrix/scalar +=, -=, *=, and /=", "[Matrix]") {
    int arrMat[4] = {1, 2, 3, 4};
    Matrix originalMat(2, 2, arrMat);
    Matrix mat1(2, 2, arrMat);

    int fac = 5;

    mat1 += fac;
    CHECK(mat1 == originalMat + fac);

    mat1 -= fac;
    CHECK(mat1 == originalMat);

    mat1 *= fac;
    CHECK(mat1 == originalMat * fac);

    mat1 /= fac;
    CHECK(mat1 == originalMat);
}