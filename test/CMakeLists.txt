find_package(GTest REQUIRED)
include(GoogleTest)
add_executable(lambda_lanczos_test lambda_lanczos_test.cpp)
target_link_libraries(lambda_lanczos_test GTest::GTest GTest::Main)
