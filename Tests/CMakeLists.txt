find_package(Catch2 3 REQUIRED)

add_executable(MatrixTestsRun MatrixTests.cpp)
add_executable(DualTestsRun DualTests.cpp)
add_executable(DatasetTestsRun DatasetTests.cpp)
add_executable(LayerTestsRun LayerTests.cpp)
add_executable(NeuralNetworkTestsRun NeuralNetworkTests.cpp)

add_executable(AllTestsRun
        MatrixTests.cpp
        DualTests.cpp
        DatasetTests.cpp
        NeuralNetworkTests.cpp
        LayerTests.cpp)



target_link_libraries(MatrixTestsRun PRIVATE Catch2::Catch2WithMain)
target_link_libraries(DualTestsRun PRIVATE Catch2::Catch2WithMain)
target_link_libraries(DatasetTestsRun PRIVATE Catch2::Catch2WithMain)
target_link_libraries(LayerTestsRun PRIVATE Catch2::Catch2WithMain)
target_link_libraries(NeuralNetworkTestsRun PRIVATE Catch2::Catch2WithMain)

target_link_libraries(AllTestsRun PRIVATE Catch2::Catch2WithMain)



include(Catch)
catch_discover_tests(MatrixTestsRun)
catch_discover_tests(DualTestsRun)
catch_discover_tests(DatasetTestsRun)
catch_discover_tests(LayerTestsRun)
catch_discover_tests(NeuralNetworkTestsRun)
include_directories(${CMAKE_SOURCE_DIR})