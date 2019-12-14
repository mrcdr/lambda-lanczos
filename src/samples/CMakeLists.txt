find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(sample1 sample1_simple.cpp)
add_executable(sample2 sample2_sparse.cpp)
add_executable(sample3 sample3_dynamic.cpp)
add_executable(sample4 sample4_use_Eigen_library.cpp)
target_link_libraries (sample4 Eigen3::Eigen)
