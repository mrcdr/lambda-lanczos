cmake_minimum_required(VERSION 3.0)
project (lambda_lanczos)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -std=c++11 -Wall -Wextra")
# The O2 flag is required to enable some warnings.

include_directories(include/lambda_lanczos)
enable_testing()
add_subdirectory(src)
add_subdirectory(test)
