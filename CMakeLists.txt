cmake_minimum_required(VERSION 3.25)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 23)

add_executable(NeuralNetwork main.cpp
        MatrixException.h
        Matrix.h
        Dual.h
        Dataset.h
        DatasetException.h
        NeuralNetwork.h
        Layer.h
        NeuralNetworkException.h
        LayerException.h)

add_subdirectory(Tests)
