find_package(GTest)
include(GoogleTest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -Wall -Wextra -Wfloat-conversion -fsanitize=address")
add_executable(lambda_lanczos_test_impl lambda_lanczos_test.cpp exponentiator_test.cpp)
add_executable(lambda_lanczos_test_lapack lambda_lanczos_test.cpp exponentiator_test.cpp)

target_compile_options(lambda_lanczos_test_lapack PRIVATE "-DLAMBDA_LANCZOS_USE_LAPACK")

target_link_libraries(lambda_lanczos_test_impl GTest::GTest GTest::Main)
target_link_libraries(lambda_lanczos_test_lapack GTest::GTest GTest::Main lapacke)

add_custom_target(lambda_lanczos_test
                    DEPENDS lambda_lanczos_test_impl
                    DEPENDS lambda_lanczos_test_lapack)

add_test(NAME FullTest COMMAND lambda_lanczos_test_impl)
add_test(NAME FullTestLapack COMMAND lambda_lanczos_test_lapack)
