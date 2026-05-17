#include <catch2/catch_test_macros.hpp>
#include <string>

#include "Dataset.h"
#include "Matrix.h"

TEST_CASE("TestData1 read", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData1.csv";;

    Dataset dataset(fileName);

    CHECK(dataset.getNumEntries() == 3);
    CHECK(dataset.getNumFields() == 3);

    string* header = dataset.getHeader();

    CHECK(header[0] == "Ones");
    CHECK(header[1] == "Twos");
    CHECK(header[2] == "Threes");

    Matrix<double> data = dataset.getData();

    CHECK(data.getRowSize() == 3);
    CHECK(data.getColSize() == 3);

    // Check that the matrix is initialized correctly
    CHECK(data.get(0, 0) == 1.0);
    CHECK(data.get(0, 1) == 2.0);
    CHECK(data.get(0, 2) == 3.0);

    // Check that dataset accessors work
    CHECK(dataset.get(0)[0] == 1.0);
    CHECK(dataset.get(0)[1] == 2.0);
    CHECK(dataset.get(0)[2] == 3.0);

}

TEST_CASE("TestData2 read", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData2.csv";;

    Dataset dataset(fileName);

    CHECK(dataset.getNumEntries() == 3);
    CHECK(dataset.getNumFields() == 2);

    string* header = dataset.getHeader();

    CHECK(header[0] == "\"Column one\"");
    CHECK(header[1] == "\"Column two\"");
}