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

    CHECK(data.getNumRows() == 3);
    CHECK(data.getNumCols() == 2);

    CHECK(dependent.getNumRows() == 3);

    // Check that the matrix is initialized correctly
    CHECK(data.get(0, 0) == 2.0);
    CHECK(data.get(0, 1) == 3.0);
    CHECK(data.get(2, 1) == 3.0);

}

TEST_CASE("TestData2 read", "[Dataset]") {  // TODO--FIX THIS!
    string fileName = "Tests/TestDatasets/TestData2.csv";;

    Dataset<double> dataset(fileName, "Column two"); //depName = "\"Column one\"" works fine, "\"Column two\"" doesn't

    CHECK(dataset.getNumEntries() == 3);
    CHECK(dataset.getNumFields() == 1);

    string* header = dataset.getHeader();

    CHECK(header[0] == "Column one");

    Matrix data = dataset.getData();
    Matrix dependent = dataset.getDependent();

    CHECK(data.getNumRows() == 3);
    CHECK(data.getNumCols() == 1);

    CHECK(dependent.getNumRows() == 3);
    CHECK(dependent.getNumCols() == 1);
}

TEST_CASE("Dataset accessors", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData1.csv";;
    Dataset<double> dataset(fileName, "Twos");

    CHECK(dataset.getValue("Ones", 0) == 1.0);
    CHECK(dataset.getValue("Ones", 1) == 1.0);
    CHECK(dataset.getValue("Ones", 2) == 1.0);
    CHECK(dataset.getValue("Threes", 2) == 3.0);

    CHECK(dataset.getDependentValue(1) == 2.0);


    CHECK(dataset.getEntry(0).get(0, 0) == 1.0);
    CHECK(dataset.getEntry(0).get(0, 1) == 3.0);

    CHECK(dataset.getColumn("Ones").get(0, 0) == 1.0);
    CHECK(dataset.getColumn("Ones").get(1, 0) == 1.0);
    CHECK(dataset.getColumn("Ones").get(2, 0) == 1.0);
}

TEST_CASE("Boston housing dataset read", "[Dataset]") {
    string fileName = "Datasets/BostonHousing.csv";
    Dataset<double> dataset(fileName, "medv");

    CHECK(dataset.getNumEntries() == 506);
    CHECK(dataset.getNumFields() == 12);
}
