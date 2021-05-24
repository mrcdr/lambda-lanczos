cmake_minimum_required(VERSION 3.0)
project (lambda_lanczos)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra")

include_directories(include/lambda_lanczos)
enable_testing()
add_subdirectory(src)
add_subdirectory(test)
