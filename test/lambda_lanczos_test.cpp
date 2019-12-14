#include <lambda_lanczos.hpp>
#include <cmath>
#include <gtest/gtest.h>

using lambda_lanczos::LambdaLanczos;

template<typename T>
using vector = std::vector<T>;

template<typename T>
using complex = std::complex<T>;

void sig_digit_test() {
  std::cout << std::endl << "-- Significant decimal digit test --" << std::endl;
  std::cout << "float " << lambda_lanczos_util::sig_decimal_digit<float>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<float>() << std::endl;
  std::cout << "double " << lambda_lanczos_util::sig_decimal_digit<double>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<double>() << std::endl;
  std::cout << "long double " << lambda_lanczos_util::sig_decimal_digit<long double>() << " "
       << lambda_lanczos_util::minimum_effective_decimal<long double>() << std::endl;
}

TEST(UNIT_TEST, INNER_PRODUCT) {
  complex<double> c1(1.0, 3.0);
  complex<double> c2(2.0, 4.0);

  vector<complex<double>> v1{3.0, c1};
  vector<complex<double>> v2{3.0, c2};

  auto result = lambda_lanczos_util::inner_prod(v1, v2);
  complex<double> correct(23.0, -2.0);

  EXPECT_DOUBLE_EQ(correct.real(), result.real());
  EXPECT_DOUBLE_EQ(correct.imag(), result.imag());
}

TEST(UNIT_TEST, L1_NORM) {
  complex<double> c1(1.0, 3.0);
  complex<double> c2(-1.0, -1.0);

  vector<complex<double>> v{c1, c2};

  EXPECT_DOUBLE_EQ(sqrt(10.0)+sqrt(2.0), lambda_lanczos_util::l1_norm(v));
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX) {
  const size_t n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
			  {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
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


  vector<double> correct_eigvec(n);
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = 1.0/sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvec[i]*engine.eps*10.0));
  }
}

TEST(DIAGONALIZE_TEST, DYNAMIC_MATRIX) {
  const size_t n = 100;

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0;i < n-1;i++) {
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

  double correct_eigvalue = -2.0*cos(M_PI/(n+1));

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_USE_COMPLEX_TYPE) {
  const size_t n = 3;
  complex<double> matrix[n][n] = { {2.0, 1.0, 1.0},
				   {1.0, 2.0, 1.0},
				   {1.0, 1.0, 2.0} };
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  auto init_vec = [&](vector<complex<double>>& vec) {
    for(size_t i = 0;i < n;i++) {
      vec[i] = complex<double>(0.0, 0.0);
    }

    vec[0] = complex<double>(0.0, 1.0);
  };

  LambdaLanczos<complex<double>> engine(matmul, n, true);
  engine.init_vector = init_vec;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  int itern  = engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec(n);
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = eigvec[0];
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), engine.eps*10.0);
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), engine.eps*10.0);
  }
}

//TEST(DIAGONALIZE_TEST, HERMITIAN_MATRIX) {
void test4() {
  const size_t n = 3;
  complex<double> matrix[n][n] = { { 0.0,  complex<double>(0.0, 1.0), 1.0},
				   { -complex<double>(0.0, 1.0), 0.0, complex<double>(0.0, 1.0) },
				   { 1.0, -complex<double>(0.0, 1.0), 0.0} };
  /* Its eigenvalues are {-2, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n);
  double eigvalue;
  vector<complex<double>> eigvec(n);
  int itern  = engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec { 1.0, complex<double>(0.0, 1.0), -1.0 };
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] *= eigvec[i];
  }
  double correct_eigvalue = -2.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), engine.eps*10.0);
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), engine.eps*10.0);
  }
}

