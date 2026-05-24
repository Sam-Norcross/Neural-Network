#include <catch2/catch_test_macros.hpp>
#include <string>

#include "Dataset.h"
#include "Matrix.h"

TEST_CASE("TestData1 read", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData1.csv";;

    Dataset<double> dataset(fileName, "Ones");

    CHECK(dataset.getNumEntries() == 3);
    CHECK(dataset.getNumFields() == 2);

    string* header = dataset.getHeader();

    CHECK(header[0] == "Twos");
    CHECK(header[1] == "Threes");
    CHECK(dataset.getDependentVar() == "Ones");

    Matrix data = dataset.getData();
    Matrix dependent = dataset.getDependent();

    CHECK(data.getRowSize() == 3);
    CHECK(data.getColSize() == 2);

    CHECK(dependent.getRowSize() == 3);

    // Check that the matrix is initialized correctly
    CHECK(data.get(0, 0) == 2.0);
    CHECK(data.get(0, 1) == 3.0);
    CHECK(data.get(2, 1) == 3.0);
}

// TEST_CASE("TestData2 read", "[Dataset]") {
//     string fileName = "Tests/TestDatasets/TestData2.csv";;
//
//     Dataset dataset(fileName);
//
//     CHECK(dataset.getNumEntries() == 3);
//     CHECK(dataset.getNumFields() == 2);
//
//     string* header = dataset.getHeader();
//
//     CHECK(header[0] == "\"Column one\"");
//     CHECK(header[1] == "\"Column two\"");
// }
//
// TEST_CASE("Dataset accessors", "[Dataset]") {
//     string fileName = "Tests/TestDatasets/TestData1.csv";;
//     Dataset dataset(fileName);
//
//     CHECK(dataset.getEntry("Ones", 0) == 1.0);
//     CHECK(dataset.getEntry("Ones", 1) == 1.0);
//     CHECK(dataset.getEntry("Ones", 2) == 1.0);
//
//     CHECK(dataset.getEntry("Twos", 1) == 2.0);
//     CHECK(dataset.getEntry("Threes", 2) == 3.0);
//
//     CHECK(dataset.getRow(0)[0] == 1.0);
//     CHECK(dataset.getRow(0)[1] == 2.0);
//     CHECK(dataset.getRow(0)[2] == 3.0);
//
//     CHECK(dataset.getColumn("Ones")[0] == 1.0);
//     CHECK(dataset.getColumn("Ones")[1] == 1.0);
//     CHECK(dataset.getColumn("Ones")[2] == 1.0);
// }
//
// TEST_CASE("Boston housing dataset read", "[Dataset]") {
//     string fileName = "Datasets/BostonHousing.csv";
//     Dataset dataset(fileName);
//
//     CHECK(dataset.getNumEntries() == 506);
//     CHECK(dataset.getNumFields() == 13);
// }
//
// TEST_CASE("Dataset/Matrix addition", "[Dataset]") {
//     string fileName = "Tests/TestDatasets/TestData1.csv";;
//     Dataset dataset(fileName);
//
//     Matrix<double> mat(dataset.getRowSize(), dataset.getColSize());
//     mat.randomize();
//
//     Matrix res = dataset + mat;
// }