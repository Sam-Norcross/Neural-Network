#include <iostream>
#include "Matrix.h"
#include "Dual.h"
#include "Dataset.h"
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

    // Dual<double> arr[4] = {Dual(1.0, 1.0), Dual(2.0, 1.0), Dual(3.0, 1.0), Dual(4.0, 1.0)};
    // Matrix mat(2, 2, arr);
    //
    // mat.display();

    // Matrix<Dual<double>> mat2(2, 2);
    //
    // mat2.display();
    //
    // Dual el1 = mat2.get(0, 0);
    // cout << el1.getValue() << ", " << el1.getDerivative() << endl;

    string fileName = "Tests/TestDatasets/TestData1.csv";
    Dataset dataset(fileName);
    cout << dataset.getNumEntries() << endl;

    return 0;
}
