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
  const int n = 8;
  double matrix[n][n] = {
      { 6, -3, -3,  0, -1,  1, -1,  1},
      {-3, -4,  2,  2, -1, -5,  0, -4},
      {-3,  2,  2, -3,  0,  0, -1, -1},
      { 0,  2, -3,  0, -3,  3,  2,  2},
      {-1, -1,  0, -3, -2,  0, -5, -4},
      { 1, -5,  0,  3,  0, -4,  5,  0},
      {-1,  0, -1,  2, -5,  5, -4,  4},
      { 1, -4, -1,  2, -4,  0,  4,  2}
  };
  /* Its eigenvalues are
   * -13.215086, -8.500332, -4.266749, -0.467272, 0.397895, 2.303837, 7.400955, 12.346751
   */

  // the matrix-vector multiplication routine
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
        out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  LambdaLanczos<double> engine(mv_mul, n, false); // true means to calculate the largest eigenvalue.

  const size_t nroot = 2;
  vector<double> eigenvalues(nroot);
  vector<vector<double>> eigenvectors(nroot, {n});
  int itern = engine.run(eigenvalues, eigenvectors);

    cout << "Iteration count: " << itern << endl;
  for (size_t iroot=0ul; iroot<nroot; ++iroot) {
      cout << "Eigen value (root "<< iroot <<"): " << setprecision(16) << eigenvalues[iroot] << endl;
      cout << "Eigen vector (root "<< iroot <<"): ";
      for (int i = 0; i < n; i++) {
          cout << eigenvectors[iroot][i] << " ";
      }
      cout << endl;
  }
  
  return EXIT_SUCCESS;
}
