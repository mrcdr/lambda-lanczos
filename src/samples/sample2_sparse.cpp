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

class IIV {
public:
  int r; // row index
  int c; // column index
  double value; // matrix element at (r,c)

  IIV(int r, int c, double value): r(r), c(c), value(value) {}
};

int main() {
  const int n = 3;
  vector<IIV> matrix;
  matrix.emplace_back(0, 1, 1.0);
  matrix.emplace_back(0, 2, 1.0);
  matrix.emplace_back(1, 0, 1.0);
  matrix.emplace_back(1, 2, -1.0);
  matrix.emplace_back(2, 0, 1.0);
  matrix.emplace_back(2, 1, -1.0);
  /*
    means a 3x3 matrix

    0  1  1
    1  0 -1
    1 -1  0 .

    Its eigenvalues are {1, 1, -2}
   */

  // the matrix-vector multiplication routine
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0; i < matrix.size(); ++i) {
      out[matrix[i].r] += matrix[i].value*in[matrix[i].c];
    }
  };

  LambdaLanczos<double> engine(mv_mul, n, false, 1); // Find 1 minimum eigenvalue
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
