find_package(Catch2 3 REQUIRED)

add_executable(MatrixTestsRun MatrixTests.cpp)

target_link_libraries(MatrixTestsRun PRIVATE Catch2::Catch2WithMain)

include(Catch)
catch_discover_tests(MatrixTestsRun)
include_directories(/Users/samnorcross/Desktop/NeuralNetwork)