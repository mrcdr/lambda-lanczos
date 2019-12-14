#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <lambda_lanczos.hpp>

using std::cout;
using std::endl;
using std::setprecision;
using lambda_lanczos::LambdaLanczos;

template<typename T>
using vector = std::vector<T>;

int main() {
  const int n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
                          {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  // the matrix-vector multiplication routine
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
        out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  LambdaLanczos<double> engine(mv_mul, n, true); // true means to calculate the largest eigenvalue.
  double eigenvalue;
  vector<double> eigenvector(n);
  int itern = engine.run(eigenvalue, eigenvector);

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigenvalue << endl;
  cout << "Eigen vector: ";
  for(int i = 0;i < n;i++) {
    cout << eigenvector[i] << " ";
  }
  cout << endl;
  
  return EXIT_SUCCESS;
}
