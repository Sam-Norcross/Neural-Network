#include <iostream>
#include "Matrix.h"
#include "Dual.h"
#include "Dataset.h"
using namespace std;



Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias);

Matrix<double> sigmoid(Matrix<double> input);
Matrix<double> sigmoidDerivative(Matrix<double> input);

Matrix<double> relu(Matrix<double> input);
Matrix<double> reluDerivative(Matrix<double> input);

double rmse(Matrix<double> actual, Matrix<double> expected);
Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected);

double relTol(double current, double previous);

int main() {
    // PROOF OF CONCEPT:
    // Simple neural network: 2 hidden layers, each with 5 nodes and a sigmoid activation function
    // RMSE cost function
    // Data are the x and y values for the function y = x //can use other functions like sin(x)

    double learningRate = 0.05;

    int nodes = 5;  // Number of nodes in each hidden layer

    // int nPts = 1;   // Number of data points
    // int numVars = 1;
    //
    // Matrix<double> data(nPts, 2);  // N x 2 (1st col is x, 2nd is y)
    //
    // for (int i = 0; i < nPts; i++) {
    //     // Could scale the x values so it samples a smaller range more densely
    //     data.get(i, 0) = i; // x val
    //     data.get(i, 1) = i; // y val
    // }
    //
    // Matrix<double> input = data.getCol(0);
    // Matrix<double> output = data.getCol(1);


    string fileName = "Datasets/BostonHousing.csv";
    Dataset<double> dataset(fileName, "medv");

    Matrix<double> input = dataset.getData().transpose();
    Matrix<double> output = dataset.getDependent().transpose();

    // input = input.getCol(0);
    // output = output.getCol(0);

    cout << "Input: " << input.dims() << endl;
    cout << "Output: " << output.dims() << endl << endl;

    int numVars = input.getNumRows();



    Matrix<double> weight1(nodes, numVars);    // 5x1 * 1x1 + 5x1    --input is a column vector
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

    int epochs = 100;
    double prevCost = 100000000;
    double cost = 1;


    // for (int i = 0; i < epochs; i++) {
    while (abs(relTol(prevCost, cost)) > 1e-8) {
        prevCost = cost;

        // Run feed forward //TODO--declare the variables outside the loop and just update them each time
        Matrix<double> z1 = feedForward(input, weight1, bias1);
        // Matrix<double> a1 = sigmoid(z1);
        Matrix<double> a1 = relu(z1);

        Matrix<double> z2 = feedForward(a1, weight2, bias2);
        // Matrix<double> a2 = sigmoid(z2);
        Matrix<double> a2 = relu(z2);

        Matrix<double> z3 = feedForward(a2, weight3, bias3);
        // Matrix<double> a3 = sigmoid(z3);
        Matrix<double> a3 = relu(z3);

        cost = rmse(a3, output);    // TODO--why is the cost always 23-24 and not closer to 0?



        // Run backpropagation

        // Partial derivatives

        // Sigmoid activation functions
        // Matrix<double> dCda3 = rmseDerivative(a3, output);
        // Matrix<double> dCdW3 = (dCda3 * sigmoidDerivative(a3)).matMul(a2.transpose());
        // Matrix<double> dCdb3 =  (dCda3 * sigmoidDerivative(a3)).sumToColVec();  // TODO--why is the sumToColVec() necessary/justified?
        //
        // Matrix<double> dCda2 = (weight3.transpose()).matMul(rmseDerivative(a3, output) * sigmoidDerivative(a3));
        // Matrix<double> dCdW2 = (dCda2 * sigmoidDerivative(a2)).matMul(a1.transpose());
        // Matrix<double> dCdb2 = (dCda2 * sigmoidDerivative(a2)).sumToColVec();
        //
        // Matrix<double> dCda1 = (weight2.transpose()).matMul(dCda2 * sigmoidDerivative(a2));
        // Matrix<double> dCdW1 = (dCda1 * sigmoidDerivative(a1)).matMul(input.transpose());
        // Matrix<double> dCdb1 = (dCda1 * sigmoidDerivative(a1)).sumToColVec();


        
        // RELU activation functions
        Matrix<double> dCda3 = rmseDerivative(a3, output);
        Matrix<double> dCdW3 = (dCda3 * reluDerivative(a3)).matMul(a2.transpose());
        Matrix<double> dCdb3 =  (dCda3 * reluDerivative(a3)).sumToColVec();  // TODO--why is the sumToColVec() necessary/justified?
        // Matrix<double> dCdb3 =  (dCda3 * reluDerivative(a3)).getCol(0);

        Matrix<double> dCda2 = (weight3.transpose()).matMul(rmseDerivative(a3, output) * reluDerivative(a3));
        Matrix<double> dCdW2 = (dCda2 * reluDerivative(a2)).matMul(a1.transpose());
        Matrix<double> dCdb2 = (dCda2 * reluDerivative(a2)).sumToColVec();
        // Matrix<double> dCdb2 = (dCda2 * reluDerivative(a2)).getCol(0);

        Matrix<double> dCda1 = (weight2.transpose()).matMul(dCda2 * reluDerivative(a2));
        Matrix<double> dCdW1 = (dCda1 * reluDerivative(a1)).matMul(input.transpose());
        Matrix<double> dCdb1 = (dCda1 * reluDerivative(a1)).sumToColVec();
        // Matrix<double> dCdb1 = (dCda1 * reluDerivative(a1)).getCol(0);


        dCdW3.display();




        // Gradient descent
        weight1 -= learningRate * dCdW1;
        bias1 -= learningRate * dCdb1;

        weight2 -= learningRate * dCdW2;
        bias2 -= learningRate * dCdb2;

        weight3 -= learningRate * dCdW3;
        bias3 -= learningRate * dCdb3;


        cout << "cost: " << cost << endl;
        cout << "prevCost: " << prevCost << endl;
        cout << "relTol(cost): " << abs(relTol(prevCost, cost)) << endl << endl;

    }

    // cout << "\n\ncost: " << cost << endl;
    // cout << "prevCost: " << prevCost << endl;
    // cout << "relTol(cost): " << abs(relTol(prevCost, cost)) << endl << endl;


    return 0;
}



Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias) {
    return weight.matMul(input).colAdd(bias);
}

// Sigmoid
Matrix<double> sigmoid(Matrix<double> input) {
    return 1.0 / (1.0 + exp(-input));
}

Matrix<double> sigmoidDerivative(Matrix<double> input) {
    return sigmoid(input) * (1.0 - sigmoid(input));
}



// RMSE
double rmse(Matrix<double> actual, Matrix<double> expected) {
    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmse()");
    }

    double numDataPts = expected.getNumCols();

    // Matrix diff = actual - expected;
    // Matrix diff2 = pow(diff, 2);

    double sum = 0;
    for (int i = 0; i < numDataPts; i++) {
        sum += pow(actual.get(0, i) - expected.get(0, i), 2);

        // sum += diff.get(0, i);
    }

    sum /= numDataPts;

    return sqrt(sum);
}

Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected) {
    double numDataPts = expected.getNumRows();
    Matrix<double> diffVec = actual - expected;

    return  diffVec / (numDataPts * rmse(actual, expected));
}



Matrix<double> relu(Matrix<double> input) {
    Matrix<double> newMat = input;

    for (int r = 0; r < input.getNumRows(); r++) {
        for (int c = 0; c < input.getNumCols(); c++) {

            if (newMat.get(r, c) < 0) {
                newMat.get(r, c) = 0;
            }

        }
    }

    return newMat;
}

Matrix<double> reluDerivative(Matrix<double> input) {
    Matrix<double> newMat = input;

    for (int r = 0; r < input.getNumRows(); r++) {
        for (int c = 0; c < input.getNumCols(); c++) {

            if (newMat.get(r, c) < 0) {
                newMat.get(r, c) = 0.0;
            } else {
                newMat.get(r, c) = 1.0;
            }

        }
    }

    return newMat;
}



double relTol(double current, double previous) {
    return (previous - current) / previous;
}