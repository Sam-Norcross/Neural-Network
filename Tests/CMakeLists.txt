find_package(Catch2 3 REQUIRED)

add_executable(MatrixTestsRun MatrixTests.cpp)
add_executable(DualTestsRun DualTests.cpp)

add_executable(AllTestsRun MatrixTests.cpp DualTests.cpp)



target_link_libraries(MatrixTestsRun PRIVATE Catch2::Catch2WithMain)
target_link_libraries(DualTestsRun PRIVATE Catch2::Catch2WithMain)

target_link_libraries(AllTestsRun PRIVATE Catch2::Catch2WithMain)



include(Catch)
catch_discover_tests(MatrixTestsRun)
catch_discover_tests(DualTestsRun)
include_directories(${CMAKE_SOURCE_DIR})