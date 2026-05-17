// TODO--Refactor this class to inherit from Matrix with a few wrapper functions to handle reading in the data--that way,
    // arithmetic will work

#include <iostream>
#include <string>
#include <fstream>

#include "DatasetException.h"
#include "Matrix.h"

#pragma once
using namespace std;

class Dataset {
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
        entries = Matrix<double>(numEntries, numFields);
        for (int entryIndex = 0; entryIndex < numEntries; entryIndex++) {
            getline(readFile, line);
            string *tokens = split(line);
            for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
                entries.get(entryIndex, fieldIndex) = stod(tokens[fieldIndex]); // stod() converts string to double
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

    Matrix<double>& getData() {
        return entries;
    }

    double get(string field, int entryIndex) {
        int fieldIndex = getFieldIndex(field);
        return entries.get(entryIndex, fieldIndex);
    }

    double* get(int entryIndex) {
        double* data = new double[numEntries];
        for (int fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
            data[fieldIndex] = entries.get(entryIndex, fieldIndex);
        }

        return data;
    }

    double* get(string field) {
        int fieldIndex = getFieldIndex(field);
        double* data = new double[numEntries];

        for (int entryIndex = 0; entryIndex < numEntries; entryIndex++) {
            data[entryIndex] =  entries.get(entryIndex, fieldIndex);
        }

        return data;
    }



private:
    string* header;     // Array with the names of each column
    int numEntries;     // Number of data points (rows in the CSV)
    int numFields;      // Number of fields (columns in the CSV
    // int* entriesInt;
    // double* entriesDouble;  // Stores the fields of each data entry
    // string* entriesString;

    Matrix<double> entries; // The rows represent individual entries, the columns represent different fields

    // Divide a comma-separated string into an array of strings
    string* split(string tokenString) {
        string* tokens = new string[numFields];

        int tokenIndex = 0;
        int startIndex = 0;
        int endIndex = 0;
        bool quotes = false; // Keeps track of when quotes open and close

        for (char c : tokenString) {
            if (c == '"') {
                quotes = !quotes;
            }
            if (c == ',' || (isspace(c) && !quotes)) { // isspace(c) checks for a newline character
                tokens[tokenIndex] = tokenString.substr(startIndex, endIndex - startIndex);

                tokenIndex++;
                startIndex = endIndex + 1;
            }
            endIndex++;
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