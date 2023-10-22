find_package(Eigen3 NO_MODULE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(sample1 sample1_simple.cpp)
add_executable(sample2 sample2_sparse.cpp)
add_executable(sample3 sample3_dynamic.cpp)
add_executable(sample4 sample4_use_Eigen_library.cpp)
add_executable(sample5 sample5_multiroot.cpp)
