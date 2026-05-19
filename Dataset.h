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
        string *headerTokens = split(line);
        header = headerTokens;

        // Iterate through data entries and store in the appropriate arrays
        setSize(numEntries, numFields);
        for (int entryIndex = 0; entryIndex < numEntries; entryIndex++) {
            getline(readFile, line);

            string *tokens = split(line);   // Split string line into an array of string tokens

            for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
                get(entryIndex, fieldIndex) = stod(tokens[fieldIndex]); // stod() converts string to double
            }
        }

        readFile.close();
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

    // Matrix<double> entries; // The rows represent individual entries, the columns represent different fields

    // Divide a comma-separated string into an array of strings
    string* split(string tokenString) {
        string* tokens = new string[numFields];

        int tokenIndex = 0;
        int startIndex = 0;
        int tokenLength = 0;
        bool quotes = false; // Keeps track of when quotes open and close

        for (char c : tokenString) {
            tokenLength++;

            if (c == '"') {
                quotes = !quotes;
            }

            int tokenStringLength = tokenString.length();

            if (c == ',' || (isspace(c) && !quotes)) {
                tokens[tokenIndex] = tokenString.substr(startIndex, tokenLength - 1);

                tokenIndex++;
                startIndex += tokenLength;
                tokenLength = 0;
            } else if (startIndex + tokenLength == tokenStringLength) {  // If the end of the string is reached, add the rest to tokens
                tokens[tokenIndex] = tokenString.substr(startIndex, tokenLength);
            }
        }

        return tokens;
    }

    int getFieldIndex(string fieldName) {
        for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
            if (header[fieldIndex] == fieldName) {
                return fieldIndex; //entries.get(fieldIndex, fieldName);
            }
        }
        throw DatasetException("Field " + fieldName + " not found");
    }

};