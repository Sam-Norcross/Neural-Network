#include <complex>
#include <iostream>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>

#include "DatasetException.h"
#include "Matrix.h"

#pragma once
using namespace std;

inline double castString(string str);

template <typename T>
T stdDev(Matrix<T> data);

template <typename T>
T mean(Matrix<T> data);



template <typename T>   // TODO--the template is currently meaningless, as all values are cast to double
class Dataset {
public:
    Dataset(string filePath, string depName) : numFields(0), numEntries(0), dependentVar(depName), dependentMean(0),
                                                dependentStdDev(0), datasetNormalized(false) {
        // Num fields starts at 0 so it doesn't count the dependent variable column

        // Open file
        ifstream readFile(filePath);
        string line;

        // Check if the file is found
        if (!readFile.is_open()) {
            throw DatasetException("Error opening file");
        }

        // Count the number of fields in the dataset
        getline(readFile, line); // Read header line
        for (char c: line) {
            if (c == ',') {
                numFields++;
            }
        }

        // Count how many data points (lines) the csv file contains
        while (!readFile.eof()) {
            getline(readFile, line);
            if (line != "" && line != "\n") {
                numEntries++;
            }
        }
        readFile.close();



        // Reopen file to read data
        readFile.open(filePath);
        getline(readFile, line);

        // Remove new line character so that files are read in the same way on both Mac and Windows
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Save header line
        header = new string[numFields];
        int fieldIndex = 0;
        int startIndex = 0;
        int fieldLength = 0;
        bool quotes = false; // Keeps track of when quotes open and close

        int depInd = -1;    // Column index of the dependent variable column

        // Read header
        string currentField = "";
        for (int i = 0; i < line.length(); i++) {
            char c = line[i];

            // If a comma of the end of the line is reached, add currentField to header
            if ((c == ',' && !quotes) || i == line.length() - 1) {

                if (i == line.length() - 1 && c != '\"') {
                    currentField += c;
                }

                // If currentField is the dependent variable, record its index
                if (currentField == depName) {
                    depInd = fieldIndex;
                    currentField = "";
                }
                else {  // Otherwise, add currentField to the header
                    header[fieldIndex] = currentField;
                    currentField = "";
                    fieldIndex++;
                }

            }
            else if (c == '\"') { // Handles quotes in the header
                quotes = !quotes;
            }
            else {
                currentField += c;
            }
        }

        // If the dependent variable wasn't found, throw an error
        if (depInd == -1) {
            throw DatasetException("The dependent variable '" + depName + "' is not listed in the .csv file");
        }

        // Iterate through data entries and store in the appropriate arrays
        data = Matrix<T>(numEntries, numFields);
        dependent = Matrix<T>(numEntries, 1);

        bool depFound = false; // Records when the dependent variable is found

        for (int lineNum = 0; lineNum < numEntries; lineNum++) {
            getline(readFile, line);

            // Remove new line character so that files are read in the same way on both Mac and Windows
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            fieldIndex = 0;
            startIndex = 0;
            fieldLength = 0;

            depFound = false;

            for (char c : line) {
                fieldLength++;
                int tokenStringLength = line.length();

                if (c == ',' || isspace(c)) {

                    //TODO--define custom casting function for more data types instead of just using stod()?

                    if (fieldIndex == depInd and depFound == false) {
                        dependent.get(lineNum, 0) = castString(line.substr(startIndex, fieldLength - 1));

                        startIndex += fieldLength;
                        fieldLength = 0;
                        depFound = true;
                    }
                    else {
                        data.get(lineNum, fieldIndex) = castString(line.substr(startIndex, fieldLength - 1));

                        fieldIndex++;
                    }

                    startIndex += fieldLength;
                    fieldLength = 0;
                } else if (startIndex + fieldLength == tokenStringLength) {
                    // If the end of the string is reached, add the rest to tokens

                    if (depFound == false) {
                        dependent.get(lineNum, 0) = castString(line.substr(startIndex, fieldLength));
                    } else {
                        data.get(lineNum, fieldIndex) = castString(line.substr(startIndex, fieldLength));
                    }
                }

            }
        }

        readFile.close();

        independentMeans = zeros<T>(1, numFields);
        independentStdDevs = zeros<T>(1, numFields);
    }

    // Constructor to create a dataset from Matrix objects for easy testing
    Dataset(Matrix<T> dataMat, Matrix<T> dependentMat) : datasetNormalized(false) {
        if (dataMat.getNumRows() != dependentMat.getNumRows()) {
            throw DatasetException("Both independent and dependent variable matrices must have the same number of rows.");
        }
        if (dependentMat.getNumCols() != 1) {
            throw DatasetException("The dependent variable matrix should only have one column");
        }

        data = dataMat;
        dependent = dependentMat;

        numEntries = dataMat.getNumRows();
        numFields = dataMat.getNumCols();

        dependentVar = "Dependent";
        header = new string[numFields];
        for (int i = 0; i < numFields; i++) {
            header[i] = "Field " + to_string(i);
        }

        dependentMean = 0;
        dependentStdDev = 0;

        independentMeans = zeros<T>(1, numFields);
        independentStdDevs = zeros<T>(1, numFields);
    }

    Dataset(string* head, string depVar, Matrix<T> dataMat, Matrix<T> dependentMat, int entries, int fields,
                bool normalized, T depMean, T depStdDev, Matrix<T> indMean, Matrix<T> indStdDev) :
                header(head), dependentVar(depVar), data(dataMat), dependent(dependentMat),
                numEntries(entries), numFields(fields), datasetNormalized(normalized), dependentMean(depMean),
                dependentStdDev(depStdDev), independentMeans(indMean), independentStdDevs(indStdDev) {}

    Dataset() : header(nullptr), dependentVar(""), numEntries(0), numFields(0), datasetNormalized(false),
                dependentMean(0), dependentStdDev(0), independentMeans(zeros<T>(1, numFields)),
                independentStdDevs(zeros<T>(1, numFields)) {}

    // Copy constructor
    Dataset(const Dataset& other) {
        dependentVar = other.dependentVar;
        data = other.data;
        dependent = other.dependent;
        numEntries = other.numEntries;
        numFields = other.numFields;

        datasetNormalized = other.datasetNormalized;

        dependentMean = other.dependentMean;
        dependentStdDev = other.dependentStdDev;

        independentMeans = other.independentMeans;
        independentStdDevs = other.independentStdDevs;

        header = new string[numFields];
        for (int i = 0; i < numFields; i++) {
            header[i] = other.header[i];
        }
    }

    ~Dataset() {
        delete [] header;
    }

    // Copy assignment operator
    Dataset& operator=(const Dataset& other) {
        if (this == &other) {
            return *this;
        }

        delete [] header;

        dependentVar = other.dependentVar;
        data = other.data;
        dependent = other.dependent;
        numEntries = other.numEntries;
        numFields = other.numFields;

        datasetNormalized = other.datasetNormalized;

        dependentMean = other.dependentMean;
        dependentStdDev = other.dependentStdDev;

        independentMeans = other.independentMeans;
        independentStdDevs = other.independentStdDevs;

        header = new string[numFields];
        for (int i = 0; i < numFields; i++) {
            header[i] = other.header[i];
        }

        return *this;
    }

    bool operator==(const Dataset& other) const {

        if (getDependentVar() != other.getDependentVar()) {
            return false;
        }

        if (numericalEquality(getData(), other.getData(), 1e-13) == false) { //if (getData() != other.getData()) {
            return false;
        }

        if (numericalEquality(getDependent(), other.getDependent()) == false) { //if (getDependent() != other.getDependent()) {
            return false;
        }

        if (getNumEntries() != other.getNumEntries()) {
            return false;
        }

        if (getNumFields() != other.getNumFields()) {
            return false;
        }

        if (getDatasetNormalized() != other.getDatasetNormalized()) {
            return false;
        }

        if (getDependentMean() != other.getDependentMean()) {
            return false;
        }

        if (getDependentStdDev() != other.getDependentStdDev()) {
            return false;
        }

        if (numericalEquality(getIndependentMeans(), other.getIndependentMeans()) == false) { //if (getIndependentMeans() != other.getIndependentMeans()) {
            return false;
        }

        if (numericalEquality(getIndependentStdDevs(), other.getIndependentStdDevs()) == false) { //if (getIndependentStdDevs() != other.getIndependentStdDevs()) {
            return false;
        }

        for (int i = 0; i < numFields; i++) {
            if (header[i] != other.header[i]) {
                return false;
            }
        }

        return true;
    }

    // Returns the number of data points (lines in the CSV)
    int getNumEntries() const {
        return numEntries;
    }

    // Returns the number of independent variables recorded for each data point (the number of columns in the CSV,
    // minus one to account for the dependent variable column)
    int getNumFields() const {
        return numFields;
    }

    string* getHeader() const {
        return header;
    }

    string getDependentVar() const {
        return dependentVar;
    }

    Matrix<T> getData() const {
        return data;
    }

    Matrix<T> getDependent() const {
        return dependent;
    }

    // Overloading these functions as non-const references allows data and dependent to be accessed and modified
    Matrix<T>& getData() {
        return data;
    }

    Matrix<T>& getDependent() {
        return dependent;
    }

    bool getDatasetNormalized() const {
        return datasetNormalized;
    }

    void setDatasetNormalized(bool normalized) {
        datasetNormalized = normalized;
    }

    T getDependentMean() const {
        return dependentMean;
    }

    T getDependentStdDev() const {
        return dependentStdDev;
    }

    Matrix<T> getIndependentMeans() const {
        return independentMeans;
    }

    Matrix<T> getIndependentStdDevs() const {
        return independentStdDevs;
    }

    Matrix<T>& getIndependentMeans() {
        return independentMeans;
    }

    Matrix<T>& getIndependentStdDevs() {
        return independentStdDevs;
    }

    double getValue(string field, int entryIndex) {
        int fieldIndex = getFieldIndex(field);
        return data.get(entryIndex, fieldIndex);
    }

    double getDependentValue(int entryIndex) {
        return dependent.get(entryIndex, 0);
    }

    Matrix<T> getEntry(int entryIndex) {
        return data.getRow(entryIndex);
    }

    Matrix<T> getColumn(int fieldIndex) {
        return data.getCol(fieldIndex);
    }

    Matrix<T> getColumn(string field) {
        int fieldIndex = getFieldIndex(field);
        return getColumn(fieldIndex);
    }

    // // Normalize each variable to have mean 0 and standard deviation 1 (Z-scores)
    // Dataset normalize() {
    //     Dataset newDataset = *this;
    //
    //     Matrix<T> col;
    //     for (int i = 0; i < newDataset.getData().getNumCols(); i++) {
    //         col = newDataset.getData().getCol(i);
    //         double colStdDev = stdDev(col);
    //
    //         if (colStdDev != 0) {
    //             newDataset.getData().setCol(i, (col - mean(col)) / colStdDev);
    //         }
    //         else {
    //             newDataset.getData().setCol(i, col - mean(col));  // Set column variable to all be 0s
    //         }
    //     }
    //
    //     T depMean = mean(dependent);
    //     T depStdDev = stdDev(dependent);
    //
    //     newDataset.setDependentMean(depMean);
    //     newDataset.setDependentStdDev(depStdDev);
    //
    //     if (depStdDev != 0) {
    //         newDataset.getDependent() = (newDataset.getDependent() - depMean) / depStdDev;
    //     }
    //     else {
    //         newDataset.getDependent() = newDataset.getDependent() - depMean;  // Set dependent variable to all be 0s
    //     }
    //
    //     return newDataset;
    // }

    // Normalize each variable to have mean 0 and standard deviation 1 (Z-scores)
    Dataset normalize() {   // TODO--make this a void function so it doesn't copy the data each time?
        Dataset newDataset = *this;

        if (getDatasetNormalized() == false) {
            Matrix<T> col;
            for (int i = 0; i < newDataset.getData().getNumCols(); i++) {
                col = newDataset.getData().getCol(i);

                double colMean = mean(col);
                double colStdDev = stdDev(col);

                newDataset.getIndependentMeans().get(0, i) = colMean;
                newDataset.getIndependentStdDevs().get(0, i) = colStdDev;

                if (colStdDev != 0) {
                    newDataset.getData().setCol(i, (col - colMean) / colStdDev);
                }
                else {
                    newDataset.getData().setCol(i, zeros<T>(numEntries, 1));  // Set column variable to all be 0s
                }
            }

            T depMean = mean(dependent);
            T depStdDev = stdDev(dependent);

            newDataset.setDependentMean(depMean);
            newDataset.setDependentStdDev(depStdDev);

            // if (depStdDev != 0) {
            //     newDataset.getDependent() = (newDataset.getDependent() - depMean) / depStdDev;
            // }
            // else {
            //     newDataset.getDependent() = newDataset.getDependent() - depMean;  // Set dependent variable to all be 0s
            // }

            newDataset.setDatasetNormalized(true);
        } else {
            cout << "Dataset is already normalized." << endl;
        }

        return newDataset;
    }

    // Apply dataset normalization to any input data of independent variables
    Matrix<T> normalizeIndependent(Matrix<T> input) {
        if (input.getNumCols() != getNumFields()) {
            throw DatasetException("Could not normalize data with  " + to_string(input.getNumCols()) +
                                    " fields when the dataset has " + to_string(getNumFields()) +
                                    " independent variables.");
        }

        Matrix<T> scaledInput = input;
        int scaledInputNumRows = scaledInput.getNumRows();

        for (int i = 0; i < input.getNumCols(); i++) {
            Matrix<T> col = input.getCol(i);
            T colStdDev = independentStdDevs.get(0, i);

            if (colStdDev != 0) {
                scaledInput.setCol(i, (col - independentMeans.get(0, i)) / colStdDev);
            }
            else {
                scaledInput.setCol(i, zeros<T>(scaledInputNumRows, 1));  // Set column variable to all be 0s
            }
        }

        return scaledInput;
    }

    Matrix<T> rescaleIndependent(Matrix<T> input) { // TODO--add a test case for this
        if (input.getNumCols() != getNumFields()) {
            throw DatasetException("Could not rescale data with  " + to_string(input.getNumCols()) +
                                    " fields when the dataset has " + to_string(getNumFields()) +
                                    " independent variables.");
        }

        Matrix<T> scaledInput = input;
        int scaledInputNumRows = scaledInput.getNumRows();

        for (int i = 0; i < input.getNumCols(); i++) {
            Matrix<T> col = input.getCol(i);
            T colStdDev = independentStdDevs.get(0, i);

            if (colStdDev != 0) {
                // scaledInput.setCol(i, (col - independentMeans.get(0, i)) / colStdDev);

                scaledInput.setCol(i, col * colStdDev + independentMeans.get(0, i));
            }
            else {
                // scaledInput.setCol(i, zeros<T>(scaledInputNumRows, 1));  // Set column variable to all be 0s

                scaledInput.setCol(i, fill<T>(scaledInputNumRows, 1, independentMeans.get(0, i)));
            }
        }

        return scaledInput;
    }

    // Reverses the normalization applied to the dependent variable to allow for meaningful NN predictions
    Matrix<T> rescaleDependent(Matrix<T> dep) {
        return dep * dependentStdDev + dependentMean;
    }

    // TODO--add better functions to normalize and de-normalize dataset and other values that can be passed in (for use in NeuralNetwork)

private:
    string* header;     // Array with the names of each column
    string dependentVar;    // Name of the dependent variable
    Matrix<T> data;     // Contains the independent variables
    Matrix<T> dependent;   // Dependent variable in the dataset
    int numEntries;     // Number of data points (rows in the CSV)
    int numFields;      // Number of fields (columns in the CSV)

    bool datasetNormalized;
    T dependentMean;
    T dependentStdDev;
    Matrix<T> independentMeans;
    Matrix<T> independentStdDevs;

    int getFieldIndex(string fieldName) {
        for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
            if (header[fieldIndex] == fieldName) {
                return fieldIndex; //entries.get(fieldIndex, fieldName);
            }
        }
        throw DatasetException("Field " + fieldName + " not found");
    }

    void setDependentMean(T depMean) {
        dependentMean = depMean;
    }

    void setDependentStdDev(T depStdDev) {
        dependentStdDev = depStdDev;
    }

};

// Functions to serialize and deserialize Dataset objects as JSON strings
template <typename T>
void to_json(nlohmann::json& j, const Dataset<T>& dataset) {

    j["header"] = nlohmann::json::array();
    string* header = dataset.getHeader();
    for (int i = 0; i < dataset.getNumFields(); i++) {
        j["header"].push_back(header[i]);
    }

    j["dependentVar"] = dataset.getDependentVar();
    j["data"] = dataset.getData();
    j["dependent"] = dataset.getDependent();
    j["numEntries"] = dataset.getNumEntries();
    j["numFields"] = dataset.getNumFields();

    j["datasetNormalized"] = dataset.getDatasetNormalized();
    j["dependentMean"] = dataset.getDependentMean();
    j["dependentStdDev"] = dataset.getDependentStdDev();
    j["indpendentMeans"] = dataset.getIndependentMeans();
    j["indpendentStdDevs"] = dataset.getIndependentStdDevs();

}

template <typename T>
void from_json(const nlohmann::json& j, Dataset<T>& dataset) {

    int numHeaderFields = j["numFields"];
    string* header = new string[numHeaderFields];
    for (int i = 0; i < j["numFields"]; i++) {
        header[i] = j.at("header").at(i);
    }

    dataset = Dataset<T>(header, j["dependentVar"], j["data"], j["dependent"], j["numEntries"],
                    j["numFields"], j["datasetNormalized"], j["dependentMean"], j["dependentStdDev"],
                    j["indpendentMeans"], j["indpendentStdDevs"]);
}




// TODO--could be overloaded to accept/return different variable types in the future?
double castString(string str) {
    try {
        return stod(str);
    }
    catch (invalid_argument& excep) {
        string errMsg1 = "The string \"";
        string errMsg2 = "\" can't be converted to double and stored in the dataset.";
        throw DatasetException(errMsg1 + str + errMsg2);
    }
}

template <typename T>
T mean(Matrix<T> data) {
    return data.sum() / data.getSize();
}

template <typename T>
T stdDev(Matrix<T> data) {
    return sqrt(pow(data - mean(data), 2).sum() / (data.getSize() - 1));   // Returns sample standard deviation
}