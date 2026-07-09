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

Matrix<double> linearAct(Matrix<double> input);
Matrix<double> linearActDerivative(Matrix<double> input);

double rmse(Matrix<double> actual, Matrix<double> expected);
Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected);

double relTol(double current, double previous);

int main() {
    // PROOF OF CONCEPT:
    // Simple neural network: 2 hidden layers, each with 5 nodes and a sigmoid activation function
    // RMSE cost function
    // Data are the x and y values for the function y = x //can use other functions like sin(x)

    double learningRate = 0.01;

    // // Number of nodes in each hidden layer
    // int nodes1 = 64;
    // int nodes2 = 32;

    // For auto-mpg_TEST dataset
    int nodes1 = 20;    // Layer 1
    int nodes2 = 10;    // Layer 2
    int nodes3 = 5;     // Layer 3



    // // TEST DATASET
    // int nPts = 50;   // Number of data points
    // int numVars = 1; // Number of input variables for each data point
    //
    // Matrix<double> data(nPts, 2);  // N x 2 (1st col is x, 2nd is y)
    //
    // for (int i = 0; i < nPts; i++) {
    //     // Could scale the x values so it samples a smaller range more densely
    //     data.get(i, 0) = i + 1; // x val
    //     data.get(i, 1) = i + 1; // y val
    // }
    //
    // Matrix<double> input = data.getCol(0).transpose();
    // Matrix<double> output = data.getCol(1).transpose();



    // // Boston Housing dataset
    // string fileName = "Datasets/BostonHousing.csv";
    // Dataset<double> datasetOriginal(fileName, "medv");

    // // Car MPG dataset
    // string fileName = "Datasets/auto-mpg.csv";
    // Dataset<double> datasetOriginal(fileName, "mpg");

    // Car MPG dataset TEST
    string fileName = "Datasets/auto-mpg_TEST.csv";
    Dataset<double> datasetOriginal(fileName, "mpg");



    Dataset dataset = datasetOriginal;//.normalize();

    Matrix<double> input = dataset.getData().transpose();
    Matrix<double> output = dataset.getDependent().transpose();

    // input = input.getCol(0);
    // output = output.getCol(0);

    int numVars = input.getNumRows();





    // Split data into training and validation data
    int numTrainSamples = 300;// 450;

    Matrix inputTrain = input.getSlice(0, input.getNumRows(), 0, numTrainSamples);
    Matrix inputValidate = input.getSlice(0, input.getNumRows(), numTrainSamples, input.getNumCols());

    Matrix outputTrain = output.getSlice(0, output.getNumRows(), 0, numTrainSamples);
    Matrix outputValidate = output.getSlice(0, output.getNumRows(), numTrainSamples, output.getNumCols());

    // If using a single data point for testing
    // Matrix inputTrain = input;
    // Matrix inputValidate = input;
    // Matrix outputTrain = output;
    // Matrix outputValidate = output;




    cout << "Input (training): " << inputTrain.dims() << endl;  // (12, 506)
    cout << "Output (training): " << outputTrain.dims() << endl << endl; // (1, 506)

    cout << "Input (validation): " << inputValidate.dims() << endl;  // (12, 506)
    cout << "Output (validation): " << outputValidate.dims() << endl << endl; // (1, 506)


    // double initVal = 1; // Bounds for weight and bias randomization
    //
    // Matrix<double> weight1(nodes1, numVars);    // 5x1 * 1x1 + 5x1    --input is a column vector
    // Matrix<double> bias1(nodes1, 1);
    // weight1.randomize(-initVal, initVal);
    // bias1.randomize(-initVal, initVal);
    //
    // Matrix<double> weight2(nodes2, nodes1);   // 5x5 * 5x1 + 5x1 = 5x1
    // Matrix<double> bias2(nodes2, 1);
    // weight2.randomize(-initVal, initVal);
    // bias2.randomize(-initVal, initVal);
    //
    // Matrix<double> weight3(1, nodes2);   // 1x5 * 5x1 + 1x1 = 1x1
    // Matrix<double> bias3(1, 1);
    // weight3.randomize(-initVal, initVal);
    // bias3.randomize(-initVal, initVal);
    //
    //
    //
    // // Initialize variables for feed forward
    // Matrix<double> z1;
    // Matrix<double> a1;
    //
    // Matrix<double> z2;
    // Matrix<double> a2;
    //
    // Matrix<double> z3;
    // Matrix<double> a3;
    //
    // Matrix<double> dCda3;
    // Matrix<double> dCdW3;
    // Matrix<double> dCdb3;
    //
    // Matrix<double> dCda2;
    // Matrix<double> dCdW2;
    // Matrix<double> dCdb2;
    //
    // Matrix<double> dCda1;
    // Matrix<double> dCdW1;
    // Matrix<double> dCdb1;



    // For auto-mpg_TEST dataset
    double initVal = 10; // Bounds for weight and bias randomization

    Matrix<double> weight1(nodes1, numVars);
    Matrix<double> bias1(nodes1, 1);
    weight1.randomize(-initVal, initVal);
    bias1.randomize(-initVal, initVal);

    Matrix<double> weight2(nodes2, nodes1);
    Matrix<double> bias2(nodes2, 1);
    weight2.randomize(-initVal, initVal);
    bias2.randomize(-initVal, initVal);

    Matrix<double> weight3(nodes3, nodes2);
    Matrix<double> bias3(nodes3, 1);
    weight3.randomize(-initVal, initVal);
    bias3.randomize(-initVal, initVal);

    Matrix<double> weight4(1, nodes3);
    Matrix<double> bias4(1, 1);
    weight4.randomize(-initVal, initVal);
    bias4.randomize(-initVal, initVal);

    // Initialize variables for feed forward
    Matrix<double> z1;
    Matrix<double> a1;
    Matrix<double> z2;
    Matrix<double> a2;
    Matrix<double> z3;
    Matrix<double> a3;
    Matrix<double> z4;
    Matrix<double> a4;
    Matrix<double> dCda4;
    Matrix<double> dCdW4;
    Matrix<double> dCdb4;
    Matrix<double> dCda3;
    Matrix<double> dCdW3;
    Matrix<double> dCdb3;
    Matrix<double> dCda2;
    Matrix<double> dCdW2;
    Matrix<double> dCdb2;
    Matrix<double> dCda1;
    Matrix<double> dCdW1;
    Matrix<double> dCdb1;



    int epochs = 50;
    double prevCost = 100;
    double cost = 1;

    // TODO--try a linear activation function for the last layer

    for (int i = 0; i < epochs; i++) {
    // while (abs(relTol(cost, prevCost)) > 1e-5) {
        prevCost = cost;

        // // Run feed forward
        // z1 = feedForward(inputTrain, weight1, bias1);
        // // a1 = sigmoid(z1);
        // a1 = relu(z1);
        //
        // z2 = feedForward(a1, weight2, bias2);
        // // a2 = sigmoid(z2);
        // a2 = relu(z2);
        //
        // z3 = feedForward(a2, weight3, bias3);
        // // a3 = sigmoid(z3);
        // a3 = relu(z3);
        // a3 = linearAct(z3);
        //
        // cost = rmse(a3, outputTrain);    // TODO--why is the cost always 23-24 and not closer to 0?


        // For auto-mpg_TEST dataset
        // Run feed forward
        z1 = feedForward(inputTrain, weight1, bias1);
        a1 = relu(z1);

        z2 = feedForward(a1, weight2, bias2);
        a2 = sigmoid(z2);

        z3 = feedForward(a2, weight3, bias3);
        a3 = relu(z3);

        z4 = feedForward(a3, weight4, bias4);
        a4 = sigmoid(z4);

        cost = rmse(a4, outputTrain);





        // Run backpropagation

        // Partial derivatives

        // // Sigmoid activation functions
        // Matrix<double> dCda3 = rmseDerivative(a3, outputTrain);
        // Matrix<double> dCdW3 = (dCda3 * sigmoidDerivative(a3)).matMul(a2.transpose());
        // Matrix<double> dCdb3 =  (dCda3 * sigmoidDerivative(a3)).sumToColVec();  // TODO--why is the sumToColVec() necessary/justified?
        //
        // Matrix<double> dCda2 = (weight3.transpose()).matMul(rmseDerivative(a3, outputTrain) * sigmoidDerivative(a3));
        // Matrix<double> dCdW2 = (dCda2 * sigmoidDerivative(a2)).matMul(a1.transpose());
        // Matrix<double> dCdb2 = (dCda2 * sigmoidDerivative(a2)).sumToColVec();
        //
        // Matrix<double> dCda1 = (weight2.transpose()).matMul(dCda2 * sigmoidDerivative(a2));
        // Matrix<double> dCdW1 = (dCda1 * sigmoidDerivative(a1)).matMul(inputTrain.transpose());
        // Matrix<double> dCdb1 = (dCda1 * sigmoidDerivative(a1)).sumToColVec();



        // // RELU activation functions
        // dCda3 = rmseDerivative(a3, outputTrain);
        // // dCdW3 = (dCda3 * reluDerivative(a3)).matMul(a2.transpose());
        // // dCdb3 =  (dCda3 * reluDerivative(a3)).sumToColVec();  // TODO--why is the sumToColVec() necessary/justified?--should it be an average instead of a sum?
        // // dCdb3 =  (dCda3 * reluDerivative(a3)).getCol(0);
        //
        // dCdW3 = (dCda3 * linearActDerivative(a3)).matMul(a2.transpose());
        // dCdb3 =  (dCda3 * linearActDerivative(a3)).sumToColVec();
        // // dCdW3 = linearActDerivative(a3).matMul(a2.transpose());
        // // dCdb3 = linearActDerivative(a3).sumToColVec();
        //
        // dCda2 = (weight3.transpose()).matMul(rmseDerivative(a3, outputTrain) * reluDerivative(a3));
        // dCdW2 = (dCda2 * reluDerivative(a2)).matMul(a1.transpose());
        // dCdb2 = (dCda2 * reluDerivative(a2)).sumToColVec();
        // // dCdb2 = (dCda2 * reluDerivative(a2)).getCol(0);
        //
        // dCda1 = (weight2.transpose()).matMul(dCda2 * reluDerivative(a2));
        // dCdW1 = (dCda1 * reluDerivative(a1)).matMul(inputTrain.transpose());
        // dCdb1 = (dCda1 * reluDerivative(a1)).sumToColVec();
        // // dCdb1 = (dCda1 * reluDerivative(a1)).getCol(0);


        // // TODO--TEST
        // dCda3 = rmseDerivative(a3, outputTrain);
        // Matrix delta3 = dCda3 * linearActDerivative(z3);
        // dCdW3 = delta3.matMul(a2.transpose());
        // dCdb3 =  delta3.sumToColVec();
        //
        // dCda2 = (weight3.transpose()).matMul(delta3);
        // Matrix delta2 = dCda2 * reluDerivative(z2);
        // dCdW2 = delta2.matMul(a1.transpose());
        // dCdb2 = delta2.sumToColVec();
        //
        // dCda1 = (weight2.transpose()).matMul(delta2);
        // Matrix delta1 = dCda1 * reluDerivative(z1);
        // dCdW1 = delta1.matMul(inputTrain.transpose());
        // dCdb1 = delta1.sumToColVec();




        // For auto-mpg_TEST dataset--TODO--compare to notes and formulas
        dCda4 = rmseDerivative(a4, outputTrain);
        Matrix delta4 = dCda4 * reluDerivative(z4);
        dCdW4 = delta4.matMul(a3.transpose());
        dCdb4 =  delta4.sumToColVec();

        dCda3 = (weight4.transpose()).matMul(delta4);
        Matrix delta3 = dCda3 * sigmoidDerivative(z3);
        dCdW3 = delta3.matMul(a2.transpose());
        dCdb3 =  delta3.sumToColVec();

        dCda2 = (weight3.transpose()).matMul(delta3);
        Matrix delta2 = dCda2 * reluDerivative(z2);
        dCdW2 = delta2.matMul(a1.transpose());
        dCdb2 = delta2.sumToColVec();

        dCda1 = (weight2.transpose()).matMul(delta2);
        Matrix delta1 = dCda1 * sigmoidDerivative(z1);
        dCdW1 = delta1.matMul(inputTrain.transpose());
        dCdb1 = delta1.sumToColVec();





        // cout << "|dCdW1|: " << sqrt(pow(dCdW1, 2).sum()) << endl;
        // cout << "|dCdb1|: " << sqrt(pow(dCdb1, 2).sum()) << endl;
        //
        // cout << "|dCdW2|: " << sqrt(pow(dCdW2, 2).sum()) << endl;
        // cout << "|dCdb2|: " << sqrt(pow(dCdb2, 2).sum()) << endl;
        //
        // cout << "|dCdW3|: " << sqrt(pow(dCdW3, 2).sum()) << endl;
        // cout << "|dCdb3|: " << sqrt(pow(dCdb3, 2).sum()) << endl;
        //
        // cout << "|delta1|: " << sqrt(pow(delta1, 2).sum()) << endl;
        // cout << "|delta2|: " << sqrt(pow(delta2, 2).sum()) << endl;
        // cout << "|delta3|: " << sqrt(pow(delta3, 2).sum()) << endl;


        // dCdW3 = (dCda3 * reluDerivative(a3)).matMul(a2.transpose());

        // cout << "dCdW3: ";
        // dCdW3.display();
        //
        // cout << "dCda3: ";
        // dCda3.display();
        //
        // cout << "reluDerivative(a3): ";
        // reluDerivative(a3).display();
        //
        // cout << "a2: ";
        // a2.display();



        // cout << "inputTrain:\n";
        // inputTrain.display();
        //
        // cout << "z1:\n";
        // z1.display();
        //
        // cout << "z2:\n";
        // z2.display();
        //
        // cout << "z3:\n";
        // z3.display();






        // // Gradient descent
        // weight1 -= learningRate * dCdW1;
        // bias1 -= learningRate * dCdb1;
        //
        // weight2 -= learningRate * dCdW2;
        // bias2 -= learningRate * dCdb2;
        //
        // weight3 -= learningRate * dCdW3;
        // bias3 -= learningRate * dCdb3;



        // For mpg-auto_TEST dataset
        // Gradient descent
        weight1 -= learningRate * dCdW1;
        bias1 -= learningRate * dCdb1;

        weight2 -= learningRate * dCdW2;
        bias2 -= learningRate * dCdb2;

        weight3 -= learningRate * dCdW3;
        bias3 -= learningRate * dCdb3;

        weight4 -= learningRate * dCdW4;
        bias4 -= learningRate * dCdb4;





        cout << "cost: " << cost << endl;
        cout << "prevCost: " << prevCost << endl;
        cout << "relTol(cost): " << abs(relTol(cost, prevCost)) << endl << endl;

    }

    // // Validate data
    // Matrix<double> z1Final = feedForward(inputValidate, weight1, bias1);
    // // Matrix<double> a1Final = sigmoid(z1Final);
    // Matrix<double> a1Final = relu(z1Final);
    //
    // Matrix<double> z2Final = feedForward(a1Final, weight2, bias2);
    // // Matrix<double> a2Final = sigmoid(z2Final);
    // Matrix<double> a2Final = relu(z2Final);
    //
    // Matrix<double> z3Final = feedForward(a2Final, weight3, bias3);
    // // Matrix<double> a3Final = sigmoid(z3Final);
    // // Matrix<double> a3Final = relu(z3Final);
    // Matrix<double> a3Final = linearAct(z3Final);
    //
    // double costValidate = rmse(a3Final, outputValidate);




    // For auto-mpg_TEST dataset
    // Validate data
    Matrix<double> z1Final = feedForward(inputValidate, weight1, bias1);
    Matrix<double> a1Final = relu(z1Final);

    Matrix<double> z2Final = feedForward(a1Final, weight2, bias2);
    Matrix<double> a2Final = sigmoid(z2Final);

    Matrix<double> z3Final = feedForward(a2Final, weight3, bias3);
    Matrix<double> a3Final = relu(z3Final);

    Matrix<double> z4Final = feedForward(a3Final, weight4, bias4);
    Matrix<double> a4Final = sigmoid(z4Final);

    double costValidate = rmse(a4Final, outputValidate);


    cout << "\n\nTraining prediction:\n";
    a3.display();
    cout << "Actual:\n";
    outputTrain.display();

    cout << "\nValidation cost: " << costValidate << endl;
    cout << "Predictions:\n";
    a3Final.display();
    cout << "Actual:\n";
    outputValidate.display();




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

    // return sqrt(pow(actual - expected, 2).sum() / numDataPts); // RMSE
    return pow(actual - expected, 2).sum() / numDataPts; // MSE
}

Matrix<double> rmseDerivative(Matrix<double> actual, Matrix<double> expected) {
    if (actual.getNumRows() != expected.getNumRows() || actual.getNumCols() != expected.getNumCols()) {
        throw MatrixException("Invalid matrix dimensions in rmseDerivative()");
    }

    double numDataPts = expected.getNumCols();

    // return  (actual - expected) / (numDataPts * rmse(actual, expected)); // RMSE
    return (actual - expected) * 2 / numDataPts; // MSE
}



Matrix<double> relu(Matrix<double> input) {
    Matrix<double> newMat = input;

    for (int r = 0; r < input.getNumRows(); r++) {
        for (int c = 0; c < input.getNumCols(); c++) {

            if (newMat.get(r, c) < 0.0) {
                newMat.get(r, c) = 0.0;
                // newMat.get(r, c) = 0.1 * input.get(r, c);   // Leaky ReLU
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
                // newMat.get(r, c) = 0.1; // Leaky ReLU
            } else {
                newMat.get(r, c) = 1.0;
            }

        }
    }

    return newMat;
}

// Linear activation function
Matrix<double> linearAct(Matrix<double> input) {
    return input;
}

Matrix<double> linearActDerivative(Matrix<double> input) {
    return ones<double>(input.getNumRows(), input.getNumCols());
}



double relTol(double current, double previous) {
    return (previous - current) / previous;
}