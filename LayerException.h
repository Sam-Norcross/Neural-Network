#include <exception>
#include <string>
#pragma once
using namespace std;

class LayerException : public exception {
public:
    LayerException(string msg) : message(msg) {}

    const char* what() const noexcept {
        return message.c_str();
    }

private:
    string message;
};
