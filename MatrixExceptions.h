//
// Created by Sam Norcross on 4/2/26.
//
#include <exception>
#include <string>
using namespace std;

class MatrixExceptions : public exception {
public:
    MatrixExceptions(string msg) : message(msg) {}

    const char* what() const noexcept {
        return message.c_str();
    }

private:
    string message;
};
