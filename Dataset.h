#include <iostream>
#include <string>
#include <fstream>

#include "DatasetException.h"
#include "Matrix.h"

#pragma once
using namespace std;

class Dataset : public Matrix<double> {
public:
    Dataset(string filePath) : numFields(1), numEntries(0) {
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

        for (char c : line) {
            fieldLength++;

            if (c == '"') {
                quotes = !quotes;
            }

            int tokenStringLength = line.length();

            if (c == ',' || (isspace(c) && !quotes)) {
                header[fieldIndex] = line.substr(startIndex, fieldLength - 1);

                fieldIndex++;
                startIndex += fieldLength;
                fieldLength = 0;
            } else if (startIndex + fieldLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens
                header[fieldIndex] = line.substr(startIndex, fieldLength);
            }
        }


        // Iterate through data entries and store in the appropriate arrays
        setSize(numEntries, numFields);

        for (int lineNum = 0; lineNum < numEntries; lineNum++) {
            getline(readFile, line);

            fieldIndex = 0;
            startIndex = 0;
            fieldLength = 0;

            for (char c : line) {
                fieldLength++;

                if (c == '"') {
                    quotes = !quotes;
                }

                int tokenStringLength = line.length();

                if (c == ',' || (isspace(c) && !quotes)) {
                    get(lineNum, fieldIndex) = stod(line.substr(startIndex, fieldLength - 1));

                    fieldIndex++;
                    startIndex += fieldLength;
                    fieldLength = 0;
                } else if (startIndex + fieldLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens
                    get(lineNum, fieldIndex) = stod(line.substr(startIndex, fieldLength));
                }

            }
        }

        readFile.close();
    }

    ~Dataset() {
        delete [] header;
        header = nullptr;
    }

    int getNumEntries() {
        return numEntries;
    }

    int getNumFields() {
        return numFields;
    }

    string* getHeader() {
        return header;
    }

    // Return underlying Matrix object
    Matrix& getData() {
        return *this;
    }

    double getEntry(string field, int entryIndex) {
        int fieldIndex = getFieldIndex(field);
        return get(entryIndex, fieldIndex);
    }

    double* getRow(int entryIndex) {
        double* data = new double[numEntries];
        for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
            data[fieldIndex] = get(entryIndex, fieldIndex);
        }

        return data;
    }

    double* getColumn(string field) {
        int fieldIndex = getFieldIndex(field);
        double* data = new double[numEntries];

        for (int entryIndex = 0; entryIndex < numEntries; entryIndex++) {
            data[entryIndex] =  get(entryIndex, fieldIndex);
        }

        return data;
    }



private:
    string* header;     // Array with the names of each column
    int numEntries;     // Number of data points (rows in the CSV)
    int numFields;      // Number of fields (columns in the CSV

    // // Divide a comma-separated string into an array of strings
    // string* split(string tokenString) {
    //     string* tokens = new string[numFields];
    //
    //     int tokenIndex = 0;
    //     int startIndex = 0;
    //     int tokenLength = 0;
    //     bool quotes = false; // Keeps track of when quotes open and close
    //
    //     for (char c : tokenString) {
    //         tokenLength++;
    //
    //         if (c == '"') {
    //             quotes = !quotes;
    //         }
    //
    //         int tokenStringLength = tokenString.length();
    //
    //         if (c == ',' || (isspace(c) && !quotes)) {
    //             tokens[tokenIndex] = tokenString.substr(startIndex, tokenLength - 1);
    //
    //             tokenIndex++;
    //             startIndex += tokenLength;
    //             tokenLength = 0;
    //         } else if (startIndex + tokenLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens
    //             tokens[tokenIndex] = tokenString.substr(startIndex, tokenLength);
    //         }
    //     }
    //
    //     return tokens;
    // }

    // // Gets the specified token from a comma-separated list of tokens
    // string getToken(string tokenString, int tokenNum) { // tokenNum starts at 0
    //     string token;
    //
    //     int tokenIndex = 0;
    //     int startIndex = 0;
    //     int tokenLength = 0;
    //     bool quotes = false; // Keeps track of when quotes open and close
    //
    //     for (char c : tokenString) {
    //         tokenLength++;
    //
    //         if (c == '"') {
    //             quotes = !quotes;
    //         }
    //
    //         int tokenStringLength = tokenString.length();
    //
    //         if (c == ',' || (isspace(c) && !quotes)) {
    //
    //             if (tokenIndex == tokenNum) {
    //                 return tokenString.substr(startIndex, tokenLength - 1);
    //             }
    //
    //             tokenIndex++;
    //             startIndex += tokenLength;
    //             tokenLength = 0;
    //         } else if (startIndex + tokenLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens
    //             if (tokenIndex == tokenNum) {
    //                 return tokenString.substr(startIndex, tokenLength);
    //             }
    //         }
    //     }
    // }

    int getFieldIndex(string fieldName) {
        for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
            if (header[fieldIndex] == fieldName) {
                return fieldIndex; //entries.get(fieldIndex, fieldName);
            }
        }
        throw DatasetException("Field " + fieldName + " not found");
    }

};