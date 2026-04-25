#include <iostream>
#include "Matrix.h"
#include <iterator>
using namespace std;

int main() {
    int arr1[4] = {1, 2, 3, 4};
    Matrix mat1(2, 2, arr1);
    // Matrix mat2(2, 2, arr1);
    // cout << (mat1 == mat2) << endl;
    //
    // int arr3[4] = {2, 2, 3, 4};
    // Matrix mat3(2, 2, arr3);
    //
    // cout << (mat1 != mat3) << endl;

    return 0;
}
