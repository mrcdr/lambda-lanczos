#include <iostream>
#include <lambda_lanczos.hpp>

template<typename T>
using real_t = lambda_lanczos::real_t<T>;


/*
 * Determine the upper bound of eigenvalue magnitudes for an n by n matrix.
 * More readable version only for type double is available below.
 */
template <typename T>
real_t<T> determine_eigenvalue_offset(const T* const* matrix, int n) {
  real_t<T> r = real_t<T>(); // Zero value of real_t<T>

  for(int i = 0;i < n;i++) {
    real_t<T> sum = real_t<T>(); // Zero value of real_t<T>
    
    for(int j = 0;j < n;j++) {
      sum += std::abs(matrix[i][j]);
    }

    if(sum > r) {
      r = sum;
    }
  }

  return r;
}



double determine_eigenvalue_offset(double** matrix, int n) {
  double r = 0.0;

  for(int i = 0;i < n;i++) {
    double sum = 0.0;
    
    for(int j = 0;j < n;j++) {
      sum += std::abs(matrix[i][j]);
    }

    if(sum > r) {
      r = sum;
    }
  }

  return r;
}







/*
 * Just for testing
 */
int main() {
  const int n = 3;
  double** matrix = new double*[n];
  for(int i = 0;i < n; i++) {
    matrix[i] = new double[n];
  }

  
  
  matrix[0][0] = -2.0; matrix[0][1] = -1.0; matrix[0][2] = -1.0;
  matrix[1][0] = -1.0; matrix[1][1] = -2.0; matrix[1][2] = -1.0;
  matrix[2][0] = -1.0; matrix[2][1] = -1.0; matrix[2][2] = -2.0;

  double r = determine_eigenvalue_offset(matrix, n);

  std::cout << "Offset should be " << r << std::endl;


  
  for(int i = 0;i < n; i++) {
    delete[] matrix[i];
  }
  delete[] matrix;
}
