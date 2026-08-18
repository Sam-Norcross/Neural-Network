#include <catch2/catch_test_macros.hpp>
#include <string>
#include <nlohmann/json.hpp>

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
    string fileName = "Tests/TestDatasets/TestData2.csv";

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

TEST_CASE("Read LinearRegression2Layer", "[Dataset]") {
    string filepath = "Tests/TestDatasets/LinearRegression2Layer.csv";
    Dataset<double> dataset(filepath, "y");

    CHECK(dataset.getNumEntries() == 20);
    CHECK(dataset.getNumFields() == 2);
}

TEST_CASE("Unknown dependent variable", "[Dataset]") {
    string fileName1 = "Tests/TestDatasets/TestData1.csv";
    string fileName2 = "Tests/TestDatasets/TestData2.csv";
    string fileName3 = "Datasets/BostonHousing.csv";
    Dataset<double> dataset;

    CHECK_THROWS(dataset = Dataset<double>(fileName1, "AAA"));
    CHECK_THROWS(dataset = Dataset<double>(fileName2, "Column four"));
    CHECK_THROWS(dataset = Dataset<double>(fileName3, "aaa"));
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

TEST_CASE("Dataset mean and stdDev helper functions", "[Dataset]") {
    double arr1[11] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    Matrix mat1(11, 1, arr1);

    CHECK(mean(mat1) == 0);
    CHECK(stdDev(mat1) == 3.3166247903554);
}

TEST_CASE("Dataset copy asignment operator", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData1.csv";
    Dataset<double> dataset1(fileName, "Ones");

    Dataset dataset2 = dataset1;

    CHECK(dataset1.getData() == dataset2.getData());
    CHECK(dataset1.getDependent() == dataset2.getDependent());

    dataset1.getData().get(0, 0) = -1;
    dataset1.getDependent().get(0, 0) = -1;

    CHECK(dataset1.getData() != dataset2.getData());
    CHECK(dataset1.getDependent() != dataset2.getDependent());

}

TEST_CASE("Dataset normalization with TestData1", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData1.csv";
    Dataset<double> dataset(fileName, "Ones");

    double arr1[3] = {0, 0, 0};
    Matrix zeros(3, 1, arr1);

    double arr2[3] = {1, 1, 1};
    Matrix onesMat(3, 1, arr2);

    Dataset newDataset = dataset.normalize();

    CHECK(newDataset.getColumn("Twos") == zeros);
    CHECK(newDataset.getColumn("Threes") == zeros);
    CHECK(newDataset.rescaleDependent(dataset.getDependent()) == onesMat);
}

TEST_CASE("Dataset normalization with TestData3", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData3.csv";
    Dataset<double> dataset(fileName, "Ones");

    Matrix dataDep = dataset.getDependent();
    Matrix dataCol1 = dataset.getColumn("Twos");
    Matrix dataCol2 = dataset.getColumn("Threes");

    Dataset dataset2 = dataset.normalize();

    // CHECK(abs(mean(dataset2.getDependent())) < 1e-16);  // Allows for roundoff error
    CHECK(abs(mean(dataset2.getColumn("Twos"))) < 1e-16);
    CHECK(abs(mean(dataset2.getColumn("Threes"))) < 1e-16);

    // CHECK(abs(1 - stdDev(dataset2.getDependent())) < 1e-16);  // Allows for roundoff error
    CHECK(abs(1 - stdDev(dataset2.getColumn("Twos"))) < 1e-16);
    CHECK(stdDev(dataset2.getColumn("Threes")) == 0);

    // Matrix rescaledDep = dataset2.rescaleDependent(dataset2.getDependent());
    // CHECK(rescaledDep == dataDep);

    Matrix zeroMat = zeros<double>(11, 1);

    CHECK(dataset2.getColumn("Threes") == zeroMat);
}

TEST_CASE("Normalizing additional data with TestData1", "[Dataset]") {
    string fileName = "Tests/TestDatasets/TestData1.csv";
    Dataset<double> dataset(fileName, "Ones");

    Matrix zerosMat = zeros<double>(3, 1);
    Matrix onesMat = ones<double>(3, 1);

    Dataset newDataset = dataset.normalize();

    CHECK(newDataset.getColumn("Twos") == zerosMat);
    CHECK(newDataset.getColumn("Threes") == zerosMat);
    CHECK(newDataset.rescaleDependent(dataset.getDependent()) == onesMat);

    double inputArr1[2] = {2, 3};
    Matrix newInd1(1, 2, inputArr1);

    CHECK(newDataset.normalizeIndependent(newInd1) == zeros<double>(1, 2));


    double inputArr2[6] = {2, 3, 2, 3, 2, 3};
    Matrix newInd2(3, 2, inputArr2);

    CHECK(newDataset.normalizeIndependent(newInd2) == zeros<double>(3, 2));
}

TEST_CASE("Normalizing more complicated generic data", "[Dataset]") {

    Matrix ind = zeros<double>(10, 3);
    Matrix dep = zeros<double>(10, 1);

    Matrix col0 = linspace<double>(0, 10, 10);
    Matrix col1 = linspace<double>(1, 1.1, 10);
    Matrix col2 = linspace<double>(10, 0, 10);

    ind.setCol(0, col0);
    ind.setCol(1, col1);
    ind.setCol(2, col2);

    dep.setCol(0, linspace<double>(-1, 9, 10));

    Dataset dataset(ind, dep);

    dataset = dataset.normalize();

    Matrix indNormalized = dataset.normalizeIndependent(ind);

    CHECK(dataset.getData() == indNormalized);

    // TODO--add tests for normalizing the dependent data?



}

TEST_CASE("Dataset equality", "[Dataset]") {
    string fileName1 = "Tests/TestDatasets/TestData1.csv";
    Dataset<double> dataset1(fileName1, "Ones");

    string fileName2 = "Tests/TestDatasets/TestData2.csv";
    Dataset<double> dataset2(fileName2, "Column two");

    Dataset dataset3 = dataset2;

    CHECK(dataset1 != dataset2);
    CHECK(dataset2 == dataset3);
}

TEST_CASE("Dataset JSON serializers", "[Dataset]") {
    string fileName = "Datasets/BostonHousing.csv";
    Dataset<double> dataset(fileName, "medv");

    dataset = dataset.normalize();

    nlohmann::json datasetJSON = dataset;

    Dataset<double> datasetFromJSON = datasetJSON.get<Dataset<double>>();

    CHECK(dataset == datasetFromJSON);
}

TEST_CASE("Dataset Matrix constructor", "[Dataset]") {
    int arrData[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int arrDep[3] = {10, 11, 12};

    Matrix data(3, 3, arrData);
    Matrix dependent(3, 1, arrDep);

    Dataset dataset(data, dependent);

    CHECK(dataset.getDependent() == dependent);
    CHECK(dataset.getData() == data);

    CHECK(dataset.getDependentVar() == "Dependent");
    CHECK(dataset.getHeader()[0] == "Field 0");
    CHECK(dataset.getHeader()[1] == "Field 1");
    CHECK(dataset.getHeader()[2] == "Field 2");

    CHECK_THROWS(dataset = Dataset(data, zeros<int>(4, 1)));
    CHECK_THROWS(dataset = Dataset(data, data));
}