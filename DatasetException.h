#include <exception>
#include <string>
#pragma once
using namespace std;

class DatasetException : public exception {
public:
    DatasetException(string msg) : message(msg) {}

    const char* what() const noexcept {
        return message.c_str();
    }

private:
    string message;
};
