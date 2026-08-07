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
    Dataset(string filePath, string depName) : numFields(0), numEntries(0), dependentVar(depName), dependentMean(0), dependentStdDev(0) {
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

        // Save header line
        header = new string[numFields];
        int fieldIndex = 0;
        int startIndex = 0;
        int fieldLength = 0;
        bool quotes = false; // Keeps track of when quotes open and close

        int depInd = -1;    // Column index of the dependent variable column

        for (char c : line) {
            fieldLength++;

            // Handles quotes in the header
            if (c == '"') {
                // Remove quotes from the string
                if (!quotes) {  // Start of the quote
                    startIndex += 1;
                }
                fieldLength -= 1;

                quotes = !quotes;
            }

            int tokenStringLength = line.length();

            if (c == ',' || (isspace(c) && !quotes)) {
                string field = line.substr(startIndex, fieldLength - 1);

                if (field == dependentVar) {
                    startIndex += fieldLength;
                    fieldLength = 0;

                    depInd = fieldIndex;
                } else {
                    header[fieldIndex] = line.substr(startIndex, fieldLength - 1);

                    fieldIndex++;
                    startIndex += fieldLength;
                    fieldLength = 0;
                }

            } else if (startIndex + fieldLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens

                if (fieldIndex < numFields) {
                    header[fieldIndex] = line.substr(startIndex, fieldLength);
                } else {
                    depInd = fieldIndex;
                }

            }
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
                        data.get(lineNum, fieldIndex) = castString(line.substr(startIndex, fieldLength - 1)); // TODO--here

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
    }

    // Constructor to create a dataset from Matrix objects for easy testing
    Dataset(Matrix<T> dataMat, Matrix<T> dependentMat) {
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
    }

    Dataset(string* head, string depVar, Matrix<T> dataMat, Matrix<T> dependentMat, int entries, int fields, T depMean, T depStdDev) :
            header(head), dependentVar(depVar), data(dataMat), dependent(dependentMat), numEntries(entries), numFields(fields),
            dependentMean(depMean), dependentStdDev(depStdDev) {}

    Dataset() : header(nullptr), dependentVar(""), numEntries(0), numFields(0), dependentMean(0), dependentStdDev(0) {}

    // Copy constructor
    Dataset(const Dataset& other) {
        dependentVar = other.dependentVar;
        data = other.data;
        dependent = other.dependent;
        numEntries = other.numEntries;
        numFields = other.numFields;

        dependentMean = other.dependentMean;
        dependentStdDev = other.dependentStdDev;

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

        dependentMean = other.dependentMean;
        dependentStdDev = other.dependentStdDev;

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

        if (getData() != other.getData()) {
            return false;
        }

        if (getDependent() != other.getDependent()) {
            return false;
        }

        if (getNumEntries() != other.getNumEntries()) {
            return false;
        }

        if (getNumFields() != other.getNumFields()) {
            return false;
        }

        if (dependentMean != other.getDependentMean()) {
            return false;
        }

        if (dependentStdDev != other.getDependentStdDev()) {
            return false;
        }

        string* otherHeader = other.getHeader();
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

    T getDependentMean() const {
        return dependentMean;
    }

    T getDependentStdDev() const {
        return dependentStdDev;
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

    // Normalize each variable to have mean 0 and standard deviation 1 (Z-scores)
    Dataset normalize() {
        Dataset newDataset = *this;

        Matrix<T> col;
        for (int i = 0; i < newDataset.getData().getNumCols(); i++) {
            col = newDataset.getData().getCol(i);
            double colStdDev = stdDev(col);

            if (colStdDev != 0) {
                newDataset.getData().setCol(i, (col - mean(col)) / colStdDev);
            }
            else {
                newDataset.getData().setCol(i, col - mean(col));  // Set column variable to all be 0s
            }
        }

        T depMean = mean(dependent);
        T depStdDev = stdDev(dependent);

        newDataset.setDependentMean(depMean);
        newDataset.setDependentStdDev(depStdDev);

        if (depStdDev != 0) {
            newDataset.getDependent() = (newDataset.getDependent() - depMean) / depStdDev;
        }
        else {
            newDataset.getDependent() = newDataset.getDependent() - depMean;  // Set dependent variable to all be 0s
        }

        return newDataset;
    }

    // Reverses the normalization applied to the dependent variable to allow for meaningful NN predictions
    Matrix<T> rescaleDependent(Matrix<T> dep) {
        return dep * dependentStdDev + dependentMean;
    }



private:
    string* header;     // Array with the names of each column
    string dependentVar;    // Name of the dependent variable
    Matrix<T> data;     // Contains the independent variables
    Matrix<T> dependent;   // Dependent variable in the dataset
    int numEntries;     // Number of data points (rows in the CSV)
    int numFields;      // Number of fields (columns in the CSV)

    T dependentMean;
    T dependentStdDev;

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


// string* header;     // Array with the names of each column
// string dependentVar;    // Name of the dependent variable
// Matrix<T> data;     // Contains the independent variables
// Matrix<T> dependent;   // Dependent variable in the dataset
// int numEntries;     // Number of data points (rows in the CSV)
// int numFields;      // Number of fields (columns in the CSV)
//
// T dependentMean;
// T dependentStdDev;

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

    j["dependentMean"] = dataset.getDependentMean();
    j["dependentStdDev"] = dataset.getDependentStdDev();

}

template <typename T>
void from_json(const nlohmann::json& j, Dataset<T>& dataset) {

    int numHeaderFields = j["numFields"];
    string* header = new string[numHeaderFields];
    for (int i = 0; i < j["numFields"]; i++) {
        header[i] = j.at("header").at(i);
    }

    dataset = Dataset<T>(header, j["dependentVar"], j["data"], j["dependent"],
                        j["numEntries"], j["numFields"],
                        j["dependentMean"], j["dependentStdDev"]);
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