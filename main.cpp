#include <iostream>
#include "Matrix.h"
#include "Dual.h"
#include "Dataset.h"
using namespace std;

Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias);
Matrix<double> activation(Matrix<double> input);
double cost(Matrix<double> actual, Matrix<double> expected);

int main() {
    // int arr1[4] = {1, 2, 3, 4};
    // Matrix mat1(2, 2, arr1);
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

    // string fileName = "Tests/TestDatasets/TestData1.csv";
    // Dataset dataset(fileName);
    // cout << dataset.getNumEntries() << endl;

    // Matrix<double> mat2(2, 2);
    // mat2.randomize();
    // mat2.display();



    // PROOF OF CONCEPT:
    // Simple neural network: 2 hidden layers, each with 5 nodes and a sigmoid activation function
    // RMSE cost function
    // Data are the x and y values for the function y = x //can use other functions like sin(x)

    int nPts = 10;   // Number of data points
    int nodes = 5;  // Number of nodes in each hidden layer

    Matrix<double> data(nPts, 2);  // N x 2 (1st col is x, 2nd is y)

    for (int i = 0; i < nPts; i++) {
        // Could scale the x values so it samples a smaller range more densely
        data.get(i, 0) = i; // x val
        data.get(i, 1) = i; // y val
    }

    Matrix<double> input = data.getCol(0);
    Matrix<double> output = data.getCol(1);

    // TEMP--use single input and output data points
    const int numPtsSmaller = 1;
    int dataPtIndex = 1;
    double inputArr[numPtsSmaller] = {input.get(dataPtIndex, 0)};
    Matrix<double> inputSmaller(numPtsSmaller, 1, inputArr);

    double outputArr[numPtsSmaller] = {input.get(dataPtIndex, 0)};
    Matrix<double> outputSmaller(numPtsSmaller, 1, outputArr);




    Matrix<double> weight1(nodes, 1);    // 5x1 * 1x1 + 5x1    --input is a column vector
    Matrix<double> bias1(nodes, 1);
    weight1.randomize();
    bias1.randomize();

    Matrix<double> weight2(numPtsSmaller, nodes);   // 5x5 * 5x1 + 5x1 = 5x1
    Matrix<double> bias2(numPtsSmaller, 1);
    weight2.randomize();
    bias2.randomize();

    // Run feed forward
    Matrix<double> intermediate = feedForward(inputSmaller, weight1, bias1);
    Matrix<double> result = feedForward(intermediate, weight2, bias2);
    double finalCost = cost(result, outputSmaller);

    cout << "COST: " << finalCost << endl;

    result.display();

    return 0;
}



Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias) {
    return weight.matMul(input) + bias;
}

Matrix<double> activation(Matrix<double> input) {
    return 1.0 / (1.0 + -1.0 * exp(input));
}

double cost(Matrix<double> actual, Matrix<double> expected) {
    double numDataPts = expected.getNumRows();

    double sum = 0;
    for (int i = 0; i < numDataPts; i++) {
        sum += pow(actual.get(i, 0) - expected.get(i, 0), 2);
    }

    sum /= numDataPts;

    return sqrt(sum);
}