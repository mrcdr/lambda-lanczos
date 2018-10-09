#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <lambda_lanczos/lambda_lanczos.hpp>

using std::cout;
using std::endl;
using std::setprecision;
using lambda_lanczos::LambdaLanczos;

template<typename T>
using vector = std::vector<T>;

template<typename T>
using complex = std::complex<T>;

void sig_digit_test() {
  cout << "-- Significant decimal digit test --" << endl;
  cout << "float " << lambda_lanczos_util::sig_decimal_digit<float>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<float>() << endl;
  cout << "double " << lambda_lanczos_util::sig_decimal_digit<double>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<double>() << endl;
  cout << "long double " << lambda_lanczos_util::sig_decimal_digit<long double>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<long double>() << endl;
}

void test1() {
  const int n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
			  {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  LambdaLanczos<double> engine(matmul, n, true);
  double eigvalue;
  vector<double> eigvec(n);
  int itern  = engine.run(eigvalue, eigvec);

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigvalue << endl;
  cout << "Eigen vector:";
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << " ";
  }
  cout << endl;
}

void test2() {
  const int n = 100;

  double offset = -10.0;
  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      out[i] += offset*in[i];
    }
    
    for(int i = 0;i < n-1;i++) {
      out[i] += -1.0*in[i+1];
      out[i+1] += -1.0*in[i];
    }

//    out[0] += -1.0*in[n-1]; // This corresponds to 
//    out[n-1] += -1.0*in[0]; // periodic boundary condition
  };
  /*
    This lambda is equivalent to applying
    following n by n matrix ("off" means "offset"):
    
    off  -1   0       ..      0
     -1 off  -1       ..      0
      0  -1 off       ..      0
      0     ..        ..      0
      0     ..      off  -1   0
      0     ..       -1 off  -1
      0     ..        0  -1 off

      Its smallest eigenvalue is -2*cos(pi/(n+1)) + offset.
      The "offset" is required to make the absolute value of
      the smallest eigenvalue maximum.
   */

  LambdaLanczos<double> engine(matmul, n);
  engine.eps = 1e-14;
  double eigvalue;
  vector<double> eigvec(n);
  int itern = engine.run(eigvalue, eigvec);  
  eigvalue -= offset;

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigvalue << endl;  
  cout << "Eigen vector:" << endl;
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << endl;
  }
  cout << endl;
}

// void test3() {
//   const int n = 3;
//   complex<double> matrix[n][n] = { {2.0, 1.0, 1.0},
// 				   {1.0, 2.0, 1.0},
// 				   {1.0, 1.0, 2.0} };
//   /* Its eigenvalues are {4, 1, 1} */

//   auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
//     for(int i = 0;i < n;i++) {
//       for(int j = 0;j < n;j++) {
// 	out[i] += matrix[i][j]*in[j];
//       }
//     } 
//   };

//   LambdaLanczos<complex<double>> engine(matmul, n, true);
//   double eigvalue;
//   vector<complex<double>> eigvec(n);
//   int itern  = engine.run(eigvalue, eigvec);

//   cout << "Iteration count: " << itern << endl;
//   cout << "Eigen value: " << setprecision(16) << eigvalue << endl;
//   cout << "Eigen vector:";
//   for(int i = 0;i < n;i++) {
//     cout << eigvec[i] << " ";
//   }
//   cout << endl;
// }

int main() {
  sig_digit_test();
  test1();
  test2();

  return EXIT_SUCCESS;
}
