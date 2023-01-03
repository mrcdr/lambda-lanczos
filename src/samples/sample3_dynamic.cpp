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
  const int n = 100;

  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0; i < n-1; ++i) {
      out[i] += -1.0*in[i+1];
      out[i+1] += -1.0*in[i];
    }
  };
  /*
    This lambda is equivalent to applying following n by n matrix

      0  -1   0       ..      0
     -1   0  -1       ..      0
      0  -1   0       ..      0
      0     ..        ..      0
      0     ..        0  -1   0
      0     ..       -1   0  -1
      0     ..        0  -1   0 .

      Its smallest eigenvalue is -2*cos(pi/(n+1)).

      (For those who are familiar with quantum physics,
      the matrix represents a Hamiltonian of a particle
      bounded by infinitely high barriers.)
   */

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
