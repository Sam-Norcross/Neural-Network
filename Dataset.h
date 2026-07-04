#include <iostream>
#include <string>
#include <fstream>

#include "DatasetException.h"
#include "Matrix.h"

#pragma once
using namespace std;

template <typename T>   // TODO--the template is currently meaningless, as all values are cast to double
class Dataset {
public:
    Dataset(string filePath, string depName) : numFields(0), numEntries(0), dependentVar(depName) {
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
                        dependent.get(lineNum, 0) = stod(line.substr(startIndex, fieldLength - 1));

                        startIndex += fieldLength;
                        fieldLength = 0;
                        depFound = true;
                    }
                    else {
                        data.get(lineNum, fieldIndex) = stod(line.substr(startIndex, fieldLength - 1)); // TODO--here

                        fieldIndex++;
                    }

                    startIndex += fieldLength;
                    fieldLength = 0;
                } else if (startIndex + fieldLength == tokenStringLength) {
                    // If the end of the string is reached, add the rest to tokens

                    if (depFound == false) {
                        dependent.get(lineNum, 0) = stod(line.substr(startIndex, fieldLength));
                    } else {
                        data.get(lineNum, fieldIndex) = stod(line.substr(startIndex, fieldLength));
                    }
                }

            }
        }

        readFile.close();
    }

    ~Dataset() {
        delete [] header;
    }

    // Returns the number of data points (lines in the CSV)
    int getNumEntries() {
        return numEntries;
    }

    // Returns the number of independent variables recorded for each data point (the number of columns in the CSV,
    // minus one to account for the dependent variable column)
    int getNumFields() {
        return numFields;
    }

    string* getHeader() {
        return header;
    }

    string getDependentVar() {
        return dependentVar;
    }

    Matrix<T> getData() {
        return data;
    }

    Matrix<T> getDependent() {
        return dependent;
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



private:
    string* header;     // Array with the names of each column
    string dependentVar;    // Name of the independent variable
    Matrix<T> data;     // Contains the independent variables
    Matrix<T> dependent;   // Dependent variable in the dataset
    int numEntries;     // Number of data points (rows in the CSV)
    int numFields;      // Number of fields (columns in the CSV)

    int getFieldIndex(string fieldName) {
        for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
            if (header[fieldIndex] == fieldName) {
                return fieldIndex; //entries.get(fieldIndex, fieldName);
            }
        }
        throw DatasetException("Field " + fieldName + " not found");
    }

};






// class Dataset : public Matrix<double> {
// public:
//     Dataset(string filePath) : numFields(1), numEntries(0) {
//         // Open file
//         ifstream readFile(filePath);
//         string line;
//
//         // Check if the file is found
//         if (!readFile.is_open()) {
//             throw DatasetException("Error opening file");
//         }
//
//         // Count the number of fields in the dataset
//         getline(readFile, line); // Read header line
//         for (char c: line) {
//             if (c == ',') {
//                 numFields++;
//             }
//         }
//
//         // Count how many data points (lines) the csv file contains
//         while (!readFile.eof()) {
//             getline(readFile, line);
//             if (line != "" && line != "\n") {
//                 numEntries++;
//             }
//         }
//         readFile.close();
//
//         // Reopen file to read data
//         readFile.open(filePath);
//         getline(readFile, line);
//
//         // Save header line
//         header = new string[numFields];
//         int fieldIndex = 0;
//         int startIndex = 0;
//         int fieldLength = 0;
//         bool quotes = false; // Keeps track of when quotes open and close
//
//         for (char c : line) {
//             fieldLength++;
//
//             if (c == '"') {
//                 quotes = !quotes;
//             }
//
//             int tokenStringLength = line.length();
//
//             if (c == ',' || (isspace(c) && !quotes)) {
//                 header[fieldIndex] = line.substr(startIndex, fieldLength - 1);
//
//                 fieldIndex++;
//                 startIndex += fieldLength;
//                 fieldLength = 0;
//             } else if (startIndex + fieldLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens
//                 header[fieldIndex] = line.substr(startIndex, fieldLength);
//             }
//         }
//
//
//         // Iterate through data entries and store in the appropriate arrays
//         setSize(numEntries, numFields);
//
//         for (int lineNum = 0; lineNum < numEntries; lineNum++) {
//             getline(readFile, line);
//
//             fieldIndex = 0;
//             startIndex = 0;
//             fieldLength = 0;
//
//             for (char c : line) {
//                 fieldLength++;
//
//                 if (c == '"') {
//                     quotes = !quotes;
//                 }
//
//                 int tokenStringLength = line.length();
//
//                 if (c == ',' || (isspace(c) && !quotes)) {
//                     get(lineNum, fieldIndex) = stod(line.substr(startIndex, fieldLength - 1));
//
//                     fieldIndex++;
//                     startIndex += fieldLength;
//                     fieldLength = 0;
//                 } else if (startIndex + fieldLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens
//                     get(lineNum, fieldIndex) = stod(line.substr(startIndex, fieldLength));
//                 }
//
//             }
//         }
//
//         readFile.close();
//     }
//
//     ~Dataset() {
//         delete [] header;
//         // header = nullptr;
//     }
//
//     int getNumEntries() {
//         return numEntries;
//     }
//
//     int getNumFields() {
//         return numFields;
//     }
//
//     string* getHeader() {
//         return header;
//     }
//
//     // Return underlying Matrix object
//     Matrix& getData() {
//         return *this;
//     }
//
//     double getEntry(string field, int entryIndex) {
//         int fieldIndex = getFieldIndex(field);
//         return get(entryIndex, fieldIndex);
//     }
//
//     double* getRow(int entryIndex) {
//         double* data = new double[numEntries];
//         for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
//             data[fieldIndex] = get(entryIndex, fieldIndex);
//         }
//
//         return data;
//     }
//
//     double* getColumn(string field) {
//         int fieldIndex = getFieldIndex(field);
//         double* data = new double[numEntries];
//
//         for (int entryIndex = 0; entryIndex < numEntries; entryIndex++) {
//             data[entryIndex] =  get(entryIndex, fieldIndex);
//         }
//
//         return data;
//     }
//
//
//
// private:
//     string* header;     // Array with the names of each column
//     int numEntries;     // Number of data points (rows in the CSV)
//     int numFields;      // Number of fields (columns in the CSV
//     Matrix<double> dependent;   // Dependent variable in the dataset
//
//     int getFieldIndex(string fieldName) {
//         for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
//             if (header[fieldIndex] == fieldName) {
//                 return fieldIndex; //entries.get(fieldIndex, fieldName);
//             }
//         }
//         throw DatasetException("Field " + fieldName + " not found");
//     }
//
// };