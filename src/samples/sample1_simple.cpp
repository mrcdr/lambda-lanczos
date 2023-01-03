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
    for(int i = 0; i < n; ++i) {
      for(int j = 0; j < n; ++j) {
        out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<double> engine(mv_mul, n, true, 1); // true means to calculate the largest eigenvalue.
  vector<double> eigenvalues;
  vector<vector<double>> eigenvectors;
  engine.run(eigenvalues, eigenvectors);

  cout << "Eigenvalue: " << setprecision(16) << eigenvalues[0] << endl;
  cout << "Eigenvector: ";
  for(int i = 0; i < n; ++i) {
    cout << eigenvectors[0][i] << " ";
  }
  cout << endl;

  return EXIT_SUCCESS;
}
