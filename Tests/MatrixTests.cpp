#include <catch2/catch_test_macros.hpp>
#include "Matrix.h"

TEST_CASE("Matrix Get", "[Matrix]") {
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

TEST_CASE("Matrix Display", "{Matrix}") {
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

TEST_CASE("Matrix Multiplication", "[Matrix]") {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);

    int arr2[4] = {2, 3, 4, 5};
    Matrix mat2(2, 2, arr2);

    int arr3[4] = {10, 13, 22, 29};
    Matrix mat3(2, 2, arr3);

    (mat1 * mat2).display();
    mat3.display();

    CHECK(mat1 * mat2 == mat3);
}