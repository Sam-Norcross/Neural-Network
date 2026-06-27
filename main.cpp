#include <iostream>
#include "Matrix.h"
#include "Dual.h"
#include "Dataset.h"
using namespace std;

Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias);
Matrix<double> sigmoid(Matrix<double> input);
Matrix<double> sigmoidDerivative(Matrix<double> input);
double rmse(Matrix<double> actual, Matrix<double> expected);
double rmseDerivative(Matrix<double> actual, Matrix<double> expected);

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

    double learningRate = 0.1;

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
    const int numPtsSmaller = 10;
    int dataPtIndex = 1;
    double inputArr[numPtsSmaller] = {input.get(dataPtIndex, 0)};
    Matrix<double> inputSmaller(1, numPtsSmaller, inputArr);

    double outputArr[numPtsSmaller] = {input.get(dataPtIndex, 0)};
    Matrix<double> outputSmaller(1, numPtsSmaller, outputArr);




    Matrix<double> weight1(nodes, 1);    // 5x1 * 1x1 + 5x1    --input is a column vector
    Matrix<double> bias1(nodes, 1);
    weight1.randomize();
    bias1.randomize();

    Matrix<double> weight2(nodes, nodes);   // 5x5 * 5x1 + 5x1 = 5x1
    Matrix<double> bias2(nodes, 1);
    weight2.randomize();
    bias2.randomize();

    Matrix<double> weight3(1, nodes);   // 1x5 * 5x1 + 1x1 = 1x1
    Matrix<double> bias3(1, 1);
    weight3.randomize();
    bias3.randomize();

    // Run feed forward
    Matrix<double> z1 = feedForward(inputSmaller, weight1, bias1);
    Matrix<double> a1 = sigmoid(z1);

    Matrix<double> z2 = feedForward(a1, weight2, bias2);
    Matrix<double> a2 = sigmoid(z2);

    Matrix<double> z3 = feedForward(a2, weight3, bias3);
    Matrix<double> a3 = sigmoid(z3);

    double finalCost = rmse(a3, outputSmaller);



    // Run backpropagation
    double rmseDerArr[1] = {rmseDerivative(a3, outputSmaller)};

    Matrix<double> dCda3 = Matrix(1, 1, rmseDerArr); //rmseDerivative(a3, outputSmaller);// * sigmoidDerivative(a3);
    // TODO--dCda3 should have dimensions of a3--rmseDerivative should return a vector

    cout << dCda3.dims() << " x " << a3.dims() << " x " << a2.transpose().dims() << endl;
    Matrix<double> dCdW3 = dCda3.matMul(sigmoidDerivative(a3).matMul(a2.transpose())); //dCdz3.matMul(a2.transpose());
    // Matrix<double> dCdW3 = (dCda3 * sigmoidDerivative(a3)).matMul(a2.transpose()); //TODO--NEW--this should be better--other derivatives will also need to be adjusted


    Matrix<double> dCdb3 =  dCda3.matMul(sigmoidDerivative(a3));

    Matrix<double> dCda2 = (weight3.transpose()).matMul(rmseDerivative(a3, outputSmaller) * sigmoidDerivative(a3));
    Matrix<double> dCdW2 = (dCda2 * sigmoidDerivative(a2).get(0, 0)).matMul(a1.transpose());
    Matrix<double> dCdb2 = dCda2 * sigmoidDerivative(a2).get(0, 0);

    Matrix<double> dCda1 = (weight2.transpose()).matMul(dCda2 * sigmoidDerivative(a2));
    Matrix<double> dCdW1 = (dCda1 * sigmoidDerivative(a1).get(0, 0)).matMul(inputSmaller.transpose());
    Matrix<double> dCdb1 = dCda1 * sigmoidDerivative(a1).get(0, 0);

    cout << "COST: " << finalCost << endl;

    return 0;
}



Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias) {
    return weight.matMul(input).colAdd(bias);

    // return weight.matMul(input) + bias;
}

// Sigmoid
Matrix<double> sigmoid(Matrix<double> input) {
    return 1.0 / (1.0 + -1.0 * exp(input));
}

Matrix<double> sigmoidDerivative(Matrix<double> input) {
    return sigmoid(input) * (1.0 - sigmoid(input));
}

// RMSE
double rmse(Matrix<double> actual, Matrix<double> expected) {
    double numDataPts = expected.getNumRows();

    double sum = 0;
    for (int i = 0; i < numDataPts; i++) {
        sum += pow(actual.get(i, 0) - expected.get(i, 0), 2);
    }

    sum /= numDataPts;

    return sqrt(sum);
}

double rmseDerivative(Matrix<double> actual, Matrix<double> expected) {
    double numDataPts = expected.getNumRows();

    double sum = 0;
    for (int i = 0; i < numDataPts; i++) {
        sum += actual.get(i, 0) - expected.get(i, 0);
    }

    sum /= numDataPts;

    return sum / rmse(actual, expected);    // TODO--should this return a vector?
}