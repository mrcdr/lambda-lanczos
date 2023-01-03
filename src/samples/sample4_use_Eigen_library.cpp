#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <lambda_lanczos.hpp>

using std::cout;
using std::endl;
using std::setprecision;
using lambda_lanczos::LambdaLanczos;

template<typename T>
using vector = std::vector<T>;

int main() {
  const int n = 3;
  Eigen::MatrixXd matrix(n, n);
  matrix <<
    2.0, 1.0, 1.0,
    1.0, 2.0, 1.0,
    1.0, 1.0, 2.0;
  /* Its eigenvalues are {4, 1, 1} */

  // the matrix-vector multiplication routine
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    auto eigen_in = Eigen::Map<const Eigen::VectorXd>(&in[0], in.size());
    auto eigen_out = Eigen::Map<Eigen::VectorXd>(&out[0], out.size());

    eigen_out = matrix * eigen_in; // Easy version
    // eigen_out.noalias() += matrix * eigen_in; // Efficient version
  };

  LambdaLanczos<double> engine(mv_mul, n, true, 1); // Find 1 maximum eigenvalue
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
