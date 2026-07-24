cmake_minimum_required(VERSION 3.25)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 23)


include(FetchContent)
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(json)



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



target_link_libraries(NeuralNetwork PRIVATE nlohmann_json::nlohmann_json)
