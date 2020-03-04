#include <lambda_lanczos.hpp>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <gtest/gtest.h>

using lambda_lanczos::LambdaLanczos;

template<typename T>
using vector = std::vector<T>;

template<typename T>
using complex = std::complex<T>;

/*void sig_digit_test() {
  std::cout << std::endl << "-- Significant decimal digit test --" << std::endl;
  std::cout << "float " << lambda_lanczos::util::sig_decimal_digit<float>() << " "
	    << lambda_lanczos::util::minimum_effective_decimal<float>() << std::endl;
  std::cout << "double " << lambda_lanczos::util::sig_decimal_digit<double>() << " "
	    << lambda_lanczos::util::minimum_effective_decimal<double>() << std::endl;
  std::cout << "long double " << lambda_lanczos::util::sig_decimal_digit<long double>() << " "
	    << lambda_lanczos::util::minimum_effective_decimal<long double>() << std::endl;
}*/

template <typename T>
void vector_initializer(vector<T>& v);

template<>
void vector_initializer(vector<double>& v) {
  std::mt19937 mt(1);
  std::uniform_real_distribution<double> rand(-1.0, 1.0);

  size_t n = v.size();
  for(size_t i = 0;i < n;i++) {
    v[i] = rand(mt);
  }
}

template<>
void vector_initializer(vector<complex<double>>& v) {
  std::mt19937 mt(1);
  std::uniform_real_distribution<double> rand(-1.0, 1.0);

  size_t n = v.size();
  for(size_t i = 0;i < n;i++) {
    v[i] = std::complex<double>(rand(mt), rand(mt));
  }
}

TEST(UNIT_TEST, INNER_PRODUCT) {
  complex<double> c1(1.0, 3.0);
  complex<double> c2(2.0, 4.0);

  vector<complex<double>> v1{3.0, c1};
  vector<complex<double>> v2{3.0, c2};

  auto result = lambda_lanczos::util::inner_prod(v1, v2);
  complex<double> correct(23.0, -2.0);

  EXPECT_DOUBLE_EQ(correct.real(), result.real());
  EXPECT_DOUBLE_EQ(correct.imag(), result.imag());
}

TEST(UNIT_TEST, L1_NORM) {
  complex<double> c1(1.0, 3.0);
  complex<double> c2(-1.0, -1.0);

  vector<complex<double>> v{c1, c2};

  EXPECT_DOUBLE_EQ(sqrt(10.0)+sqrt(2.0), lambda_lanczos::util::l1_norm(v));
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

  LambdaLanczos<double> engine(matmul, n, true);
  engine.init_vector = vector_initializer<double>;
  engine.eigenvalue_offset = 6.0;

  double eigvalue;
  vector<double> eigvec(1); // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);


  vector<double> correct_eigvec(n);
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = 1.0/sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_NOT_FIX_RANDOM_SEED) {
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

  LambdaLanczos<double> engine(matmul, n, true);
  engine.eigenvalue_offset = 6.0;

  double eigvalue;
  vector<double> eigvec(1); // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);


  vector<double> correct_eigvec(n);
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = 1.0/sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, DYNAMIC_MATRIX) {
  const size_t n = 10;

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
  engine.init_vector = vector_initializer<double>;
  engine.eps = 1e-14;
  engine.eigenvalue_offset = -10.0;
  double eigvalue;
  vector<double> eigvec(n);
  engine.run(eigvalue, eigvec);

  double correct_eigvalue = -2.0*cos(M_PI/(n+1));
  auto sign = eigvec[0]/std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for(size_t i = 0;i < n;i++) {
    correct_eigvec[i] = sign * std::sin((i+1)*M_PI/(n+1));
  }
  lambda_lanczos::util::normalize(correct_eigvec);

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }
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

  LambdaLanczos<complex<double>> engine(matmul, n, true);
  engine.init_vector = vector_initializer<complex<double>>;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec(n);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0)*std::arg(eigvec[0]));
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = 1.0 / std::sqrt(n) * phase_factor;
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue*engine.eps*10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_USE_COMPLEX_TYPE_NOT_FIX_RANDOM_SEED) {
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

  LambdaLanczos<complex<double>> engine(matmul, n, true);
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec(n);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0)*std::arg(eigvec[0]));
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] = 1.0 / std::sqrt(n) * phase_factor;
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue*engine.eps*10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue*engine.eps*10));
  }
}

TEST(DIAGONALIZE_TEST, HERMITIAN_MATRIX) {
  const size_t n = 3;
  const auto I_ = complex<double>(0.0, 1.0);
  complex<double> matrix[n][n] = { { 0.0, I_  , 1.0},
				   { -I_, 0.0 , I_ },
				   { 1.0, -I_ , 0.0} };
  /* Its eigenvalues are {-2, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n);
  engine.init_vector = vector_initializer<complex<double>>;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);


  vector<complex<double>> correct_eigvec { 1.0, I_, -1.0 };
  lambda_lanczos::util::normalize(correct_eigvec);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0)*std::arg(eigvec[0]));
  for(size_t i = 0;i < n; i++) {
    correct_eigvec[i] *= phase_factor;
  }

  double correct_eigvalue = -2.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue*engine.eps*10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue*engine.eps*10));
  }
}

template <typename T, typename RE>
void generate_random_matrix(T** a, vector<T>& eigvec, T& eigvalue,
			    size_t n, size_t rand_n, RE eng) {
  const T min_eigvalue = 1.0;
  std::uniform_int_distribution<size_t> dist_index(0, n-1);
  std::uniform_real_distribution<double> dist_angle(0.0, 2*M_PI);
  std::uniform_real_distribution<double> dist_element(min_eigvalue, n*10);

  T max_eigvalue = min_eigvalue;
  size_t max_eig_index = 0;
  for(size_t i = 0;i < n;i++) {
    a[i] = a[0]+n*i;
    std::fill(a[i], a[i]+n, 0.0);
    a[i][i] = dist_element(eng);
    if(a[i][i] > max_eigvalue) {
      max_eigvalue = a[i][i];
      max_eig_index = i;
    }
  }

  eigvalue = max_eigvalue;

  /* Eigenvector corresponding to the maximum eigenvalue */
  std::fill(eigvec.begin(), eigvec.end(), T());
  eigvec[max_eig_index] = 1.0;

  for(size_t i = 0;i < rand_n;i++) {
    size_t k = dist_index(eng);
    size_t l = dist_index(eng);
    while(k == l) {
      l = dist_index(eng);
    }

    T theta = dist_angle(eng);

    T c = std::cos(theta);
    T s = std::sin(theta);
    T akk = a[k][k];
    T akl = a[k][l];
    T all = a[l][l];

    for(size_t i = 0;i < n;i++) {
      T aki_next = c*a[k][i] - s*a[l][i];
      a[l][i]    = s*a[k][i] + c*a[l][i];
      a[k][i] = aki_next;
    }

    /* Symmetrize */
    for(size_t i = 0;i < n;i++) {
      a[i][k] = a[k][i];
      a[i][l] = a[l][i];
    }

    a[k][k] = c*(c*akk - s*akl) - s*(c*akl - s*all);
    a[k][l] = s*(c*akk - s*akl) + c*(c*akl - s*all);
    a[l][k] = a[k][l];
    a[l][l] = s*(s*akk + c*akl) + c*(s*akl + c*all);


    T vk_next = c*eigvec[k] - s*eigvec[l];
    eigvec[l] = s*eigvec[k] + c*eigvec[l];
    eigvec[k] = vk_next;
  }
}

TEST(DIAGONALIZE_TEST, RANDOM_SYMMETRIC_MATRIX) {
  const size_t n = 10;

  double** matrix = new double*[n];
  matrix[0] = new double[n*n];
  for(size_t i = 0;i < n;i++) {
    matrix[i] = matrix[0]+n*i;
  }

  vector<double> correct_eigvec(n);
  double correct_eigvalue = 0.0;

  generate_random_matrix(matrix, correct_eigvec, correct_eigvalue,
			 n, n*10, std::mt19937(1));

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0;i < n;i++) {
      for(size_t j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true);
  engine.init_vector = vector_initializer<double>;
  engine.eps = 1e-14;
  double eigvalue;
  vector<double> eigvec(n);
  engine.run(eigvalue, eigvec);

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue*engine.eps));
  auto sign = eigvec[0]/std::abs(eigvec[0]);
  for(size_t i = 0;i < n;i++) {
    EXPECT_NEAR(correct_eigvec[i]*sign, eigvec[i], std::abs(correct_eigvalue*engine.eps*10));
  }

  delete[] matrix[0];
  delete[] matrix;
}

