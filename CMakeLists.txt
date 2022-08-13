cmake_minimum_required(VERSION 3.0)
project (lambda_lanczos)

include_directories(include/lambda_lanczos)
enable_testing()
add_subdirectory(src)
add_subdirectory(test)
