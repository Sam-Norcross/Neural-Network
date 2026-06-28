#include <iostream>
#include "Matrix.h"
#include "Dual.h"
#include "Dataset.h"
using namespace std;

Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias);
Matrix<double> sigmoid(Matrix<double> input);
Matrix<double> sigmoidDerivative(Matrix<double> input);
double rmse(Matrix<double> actual, Matrix<double> expected);
Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected);

int main() {
    // PROOF OF CONCEPT:
    // Simple neural network: 2 hidden layers, each with 5 nodes and a sigmoid activation function
    // RMSE cost function
    // Data are the x and y values for the function y = x //can use other functions like sin(x)

    double learningRate = 0.1;

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

    // cout << input.dims() << endl;
    // cout << output.dims() << endl;

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

    int epochs = 10;
    for (int i = 0; i < epochs; i++) {

        // Run feed forward
        Matrix<double> z1 = feedForward(input, weight1, bias1);
        Matrix<double> a1 = sigmoid(z1);

        Matrix<double> z2 = feedForward(a1, weight2, bias2);
        Matrix<double> a2 = sigmoid(z2);

        Matrix<double> z3 = feedForward(a2, weight3, bias3);
        Matrix<double> a3 = sigmoid(z3);

        double finalCost = rmse(a3, output);



        // Run backpropagation

        // Partial derivatives
        Matrix<double> dCda3 = rmseDerivative(a3, output);
        Matrix<double> dCdW3 = (dCda3 * sigmoidDerivative(a3)).matMul(a2.transpose());
        Matrix<double> dCdb3 =  (dCda3 * sigmoidDerivative(a3)).sumToColVec();  // TODO--why is the sumToColVec() necessary/justified?

        Matrix<double> dCda2 = (weight3.transpose()).matMul(rmseDerivative(a3, output) * sigmoidDerivative(a3));
        Matrix<double> dCdW2 = (dCda2 * sigmoidDerivative(a2)).matMul(a1.transpose());
        Matrix<double> dCdb2 = (dCda2 * sigmoidDerivative(a2)).sumToColVec();

        Matrix<double> dCda1 = (weight2.transpose()).matMul(dCda2 * sigmoidDerivative(a2));
        Matrix<double> dCdW1 = (dCda1 * sigmoidDerivative(a1)).matMul(input.transpose());
        Matrix<double> dCdb1 = (dCda1 * sigmoidDerivative(a1)).sumToColVec();


        cout << "\nW3\n";
        weight3.display();
        cout << "\ndCdW3\n";
        dCdW3.display();

        cout << "\ndCda3\n";
        dCda3.display();


        
        // cout << "\nW1\n";
        // weight1.display();
        // cout << "\ndCdW1\n";
        // dCdW1.display();
        //
        // cout << "\ndCda1\n";
        // dCda1.display();


        // Gradient descent
        weight1 -= learningRate * dCdW1;
        bias1 -= learningRate * dCdb1;

        weight2 -= learningRate * dCdW2;
        bias2 -= learningRate * dCdb2;

        weight3 -= learningRate * dCdW3;
        bias3 -= learningRate * dCdb3;

        // cout << "\nW1\n";
        // weight1.display();

        // cout << "\nB1\n";
        // bias1.display();
        // cout << "\ndCdB1\n";
        // dCdb1.display();

        // cout << "\nA1\n";
        // a1.display();
        //
        // cout << "\nA2\n";
        // a2.display();
        //
        // cout << "\nA3\n";
        // a3.display();


        // weight1 -= learningRate;// * dCdW1;
        // bias1 -= learningRate;// * dCdb1;
        //
        // weight2 -= learningRate;// * dCdW2;
        // bias2 -= learningRate;// * dCdb2;
        //
        // weight3 -= learningRate;// * dCdW3;
        // bias3 -= learningRate;// * dCdb3;



        // cout << "dCda3: " << dCda3.dims() << ", " << a3.dims() << endl;
        // cout << "dCdW3: " << dCdW3.dims() << ", " << weight3.dims() << endl;
        // cout << "dCdb3: " << dCdb3.dims() << ", " << bias3.dims() << endl << endl;
        //
        // cout << "dCda2: " << dCda2.dims() << ", " << a2.dims() << endl;
        // cout << "dCdW2: " << dCdW2.dims() << ", " << weight2.dims() << endl;
        // cout << "dCdb2: " << dCdb2.dims() << ", " << bias2.dims() << endl << endl;
        //
        // cout << "dCda1: " << dCda1.dims() << ", " << a1.dims() << endl;
        // cout << "dCdW1: " << dCdW1.dims() << ", " << weight1.dims() << endl;
        // cout << "dCdb1: " << dCdb1.dims() << ", " << bias1.dims() << endl << endl;

        cout << "COST: " << finalCost << endl;
    }
    return 0;
}



Matrix<double> feedForward(Matrix<double> input, Matrix<double> weight, Matrix<double> bias) {
    return weight.matMul(input).colAdd(bias);
}

// Sigmoid
Matrix<double> sigmoid(Matrix<double> input) {
    return 1.0 / (1.0 + -1.0 * exp(input));// + 0.001; // TODO--added 0.001 to deal with NaN values from everything being mapped to 0
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

Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected) {
    double numDataPts = expected.getNumRows();
    Matrix<double> diffVec = actual - expected;

    return  diffVec / (numDataPts * rmse(actual, expected));
}
