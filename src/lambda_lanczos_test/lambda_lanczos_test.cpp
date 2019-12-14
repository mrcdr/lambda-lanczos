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

template<typename T>
using complex = std::complex<T>;

void sig_digit_test() {
  cout << endl << "-- Significant decimal digit test --" << endl;
  cout << "float " << lambda_lanczos_util::sig_decimal_digit<float>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<float>() << endl;
  cout << "double " << lambda_lanczos_util::sig_decimal_digit<double>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<double>() << endl;
  cout << "long double " << lambda_lanczos_util::sig_decimal_digit<long double>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<long double>() << endl;
}

void inner_prod_test() {
  cout << endl << "-- Inner product test --" << endl;
  complex<double> c1(1.0, 3.0);
  complex<double> c2(2.0, 4.0);

  vector<complex<double>> v1{3.0, c1};
  vector<complex<double>> v2{3.0, c2};

  // (3, 1+3i)*(3, 2+4i) = 3*3 + (1-3i)*(2+4i) = 9 + 2 + 12 - 2i = 23 - 2i
  
  cout << "ans= " << lambda_lanczos_util::inner_prod(v1, v2) << endl;
}

void l1_norm_test() {
  cout << endl << "-- L1 norm test --" << endl;
  complex<double> c1(1.0, 3.0);
  complex<double> c2(-1.0, -1.0);

  vector<complex<double>> v{c1, c2};
  // sqrt(10) + sqrt(2) = 4.576...

  cout << "ans= " << lambda_lanczos_util::l1_norm(v) << endl;
}

void test1() {
  cout << endl << "-- Diagonalization test (real symmetric)  --" << endl;
  
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

  auto init_vec = [&](vector<double>& vec) {
    for(int i = 0;i < n;i++) {
      vec[i] = 0.0;
    }

    vec[0] = 1.0;
  };

  LambdaLanczos<double> engine(matmul, n, true);
  engine.init_vector = init_vec;
  engine.eigenvalue_offset = 6.0;
  
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
  cout << endl << "-- Diagonalization test (real symmetric, large)  --" << endl;
  
  const int n = 100;
    
  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n-1;i++) {
      out[i] += -1.0*in[i+1];
      out[i+1] += -1.0*in[i];
    }

//    out[0] += -1.0*in[n-1]; // This corresponds to 
//    out[n-1] += -1.0*in[0]; // periodic boundary condition
  };
  /*
    This lambda is equivalent to applying following n by n matrix
    
      0  -1   0       ..      0
     -1   0  -1       ..      0
      0  -1   0       ..      0
      0     ..        ..      0
      0     ..        0  -1   0
      0     ..       -1   0  -1
      0     ..        0  -1   0

      Its smallest eigenvalue is -2*cos(pi/(n+1)).
   */

  LambdaLanczos<double> engine(matmul, n);  
  engine.eps = 1e-14;
  engine.eigenvalue_offset = -10.0;  
  double eigvalue;
  vector<double> eigvec(n);
  int itern = engine.run(eigvalue, eigvec);

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigvalue << endl;  
  cout << "Eigen vector:" << endl;
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << endl;
  }
  cout << endl;
}

void test3() {
  cout << endl << "-- Diagonalization test (real symmetric, but using complex type)  --" << endl;
  
  const int n = 3;
  complex<double> matrix[n][n] = { {2.0, 1.0, 1.0},
				   {1.0, 2.0, 1.0},
				   {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  auto init_vec = [&](vector<complex<double>>& vec) {
    for(int i = 0;i < n;i++) {
      vec[i] = complex<double>(0.0, 0.0);
    }

    vec[0] = complex<double>(0.0, 1.0);
  };

  LambdaLanczos<complex<double>> engine(matmul, n, true);
  engine.init_vector = init_vec;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  int itern  = engine.run(eigvalue, eigvec);

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigvalue << endl;
  cout << "Eigen vector:";
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << " ";
  }
  cout << endl;
}

void test4() {
  cout << endl << "-- Diagonalization test (complex Hermitian)  --" << endl;
  
  using namespace std::complex_literals;
  
  const int n = 3;
  complex<double> matrix[n][n] = { { 0.0,  1i, 1.0},
				   { -1i, 0.0,  1i},
				   { 1.0, -1i, 0.0} };
  /* Its eigenvalues are {-2, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  LambdaLanczos<complex<double>> engine(matmul, n);
  double eigvalue;
  vector<complex<double>> eigvec(n);
  int itern  = engine.run(eigvalue, eigvec);

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigvalue << endl;
  cout << "Eigen vector:";
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << " ";
  }
  cout << endl;
}

int main() {
  sig_digit_test();
  inner_prod_test();
  l1_norm_test();
  test1();
  test2();
  test3();
  test4();

  return EXIT_SUCCESS;
}
