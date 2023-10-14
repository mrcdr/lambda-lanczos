find_package(GTest)
include(GoogleTest)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -O0 -Wall -Wextra -Wfloat-conversion -fsanitize=address")

add_executable(lambda_lanczos_test_impl lambda_lanczos_test.cpp exponentiator_test.cpp)
add_executable(lambda_lanczos_test_lapack lambda_lanczos_test.cpp exponentiator_test.cpp)
# add_executable(lambda_lanczos_test_mkl lambda_lanczos_test.cpp exponentiator_test.cpp)

target_compile_options(lambda_lanczos_test_lapack PRIVATE "-DLAMBDA_LANCZOS_USE_LAPACK")
# target_compile_options(lambda_lanczos_test_mkl PRIVATE "-DLAMBDA_LANCZOS_USE_MKL")

# target_include_directories(lambda_lanczos_test_mkl PRIVATE ${MKLROOT}/include)
# target_link_directories(lambda_lanczos_test_mkl PRIVATE ${MKLROOT}/lib)

target_link_libraries(lambda_lanczos_test_impl GTest::GTest GTest::Main)
target_link_libraries(lambda_lanczos_test_lapack GTest::GTest GTest::Main lapacke)

file(READ "/etc/issue" ETC_ISSUE)
string(REGEX MATCH "Ubuntu|Debian" DIST ${ETC_ISSUE})
if(DIST STREQUAL "")
  target_link_libraries(lambda_lanczos_test_lapack cblas)
else()
  message(STATUS "Ubuntu or Debian detected. blas will be linked for cblas functions.")
  target_link_libraries(lambda_lanczos_test_lapack blas)
endif()

# target_link_libraries(lambda_lanczos_test_mkl GTest::GTest GTest::Main -Wl,--start-group libmkl_intel_lp64.a libmkl_sequential.a libmkl_core.a -Wl,--end-group)

add_custom_target(lambda_lanczos_test
                    DEPENDS lambda_lanczos_test_impl
                    DEPENDS lambda_lanczos_test_lapack
                    # DEPENDS lambda_lanczos_test_mkl
)

add_test(NAME FullTest COMMAND lambda_lanczos_test_impl)
add_test(NAME FullTestLapack COMMAND lambda_lanczos_test_lapack)
# add_test(NAME FullTestMKL COMMAND lambda_lanczos_test_mkl)
