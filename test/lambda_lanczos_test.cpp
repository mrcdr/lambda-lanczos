#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <iostream>
#include <lambda_lanczos.hpp>
#include <lambda_lanczos_tridiagonal_impl.hpp>
#include <string>

using lambda_lanczos::LambdaLanczos;

template <typename T>
using vector = std::vector<T>;

template <typename T>
using complex = std::complex<T>;

template <typename T>
void vector_initializer(vector<T>& v);

template <>
void vector_initializer(vector<double>& v) {
  std::mt19937 mt(1);
  std::uniform_real_distribution<double> rand(-1.0, 1.0);

  size_t n = v.size();
  for (size_t i = 0; i < n; ++i) {
    v[i] = rand(mt);
  }
}

template <>
void vector_initializer(vector<complex<double>>& v) {
  std::mt19937 mt(1);
  std::uniform_real_distribution<double> rand(-1.0, 1.0);

  size_t n = v.size();
  for (size_t i = 0; i < n; ++i) {
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

TEST(UNIT_TEST, SCHMIDT_ORTHOGONALIZATION) {
  const size_t n = 10;

  std::mt19937 eng(1);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  const size_t num_vec = n / 2;
  vector<vector<complex<double>>> us;
  for (size_t k = 0; k < num_vec; ++k) {
    vector<complex<double>> u(n);
    for (auto& elem : u) {
      elem = complex<double>(dist(eng), dist(eng));
    }

    lambda_lanczos::util::schmidt_orth(u, us.begin(), us.end());
    lambda_lanczos::util::normalize(u);
    us.push_back(u);
  }

  vector<complex<double>> v(n);
  for (auto& elem : v) {
    elem = complex<double>(dist(eng), dist(eng));
  }
  lambda_lanczos::util::schmidt_orth(v, us.begin(), us.end());

  for (const auto& u : us) {
    auto ip = lambda_lanczos::util::inner_prod(v, u);
    EXPECT_NEAR(0.0, ip.real(), 1e-15 * n);
    EXPECT_NEAR(0.0, ip.imag(), 1e-15 * n);
  }
}

TEST(UNIT_TEST, MANHATTAN_NORM) {
  complex<double> c1(1.0, 3.0);
  complex<double> c2(-1.0, -1.0);

  vector<complex<double>> v{c1, c2};

  EXPECT_DOUBLE_EQ(1.0 + 3.0 + 1.0 + 1.0, lambda_lanczos::util::m_norm(v));
}

TEST(UNIT_TEST, SORT_EIGENPAIRS) {
  vector<double> eigvals{2, -1, 0};
  vector<vector<complex<double>>> eigvecs{{2, 2, 2}, {0, 0, 0}, {1, 1, 1}};
  const size_t n = eigvals.size();

  lambda_lanczos::util::sort_eigenpairs(eigvals, eigvecs, true);

  const vector<double> expected_eigvals{-1, 0, 2};
  const vector<vector<complex<double>>> expected_eigvecs{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}};

  for (size_t i = 0; i < n; ++i) {
    EXPECT_DOUBLE_EQ(expected_eigvals[i], eigvals[i]);
    for (size_t j = 0; j < n; ++j) {
      EXPECT_DOUBLE_EQ(std::real(expected_eigvecs[i][j]), std::real(eigvecs[i][j]));
    }
  }
}

TEST(UNIT_TEST, STRINGIFY) {
  vector<double> v{1, 2, 3};
  std::string str = lambda_lanczos::util::vectorToString(v);
  std::cout << str << std::endl;

  EXPECT_STREQ("1 2 3", str.c_str());
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX) {
  const size_t n = 3;
  double matrix[n][n] = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true, 1);
  engine.init_vector = vector_initializer<double>;
  engine.eigenvalue_offset = 6.0;

  double eigvalue;
  vector<double> eigvec(1);  // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);

  auto sign = eigvec[0] / std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] = sign * 1.0 / sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_FLOAT) {
  const size_t n = 3;
  float matrix[n][n] = {{2.0f, 1.0f, 1.0f}, {1.0f, 2.0f, 1.0f}, {1.0f, 1.0f, 2.0f}};
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<float>& in, vector<float>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<float> engine(matmul, n, true, 1);

  float eigvalue;
  vector<float> eigvec(1);  // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);

  auto sign = eigvec[0] / std::abs(eigvec[0]);
  vector<float> correct_eigvec(n);
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] = sign * 1.0f / std::sqrt(3.0f);
  }
  float correct_eigvalue = 4.0f;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_MULTIPLE_VALUE_RETURN_FEATURE) {
  const size_t n = 3;
  double matrix[n][n] = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true, 1);
  engine.init_vector = vector_initializer<double>;
  engine.eigenvalue_offset = 6.0;

  auto [eigenvalues, eigenvectors] = engine.run();  // C++17 multiple value return
  auto eigvalue = eigenvalues[0];
  auto eigvec = eigenvectors[0];

  auto sign = eigvec[0] / std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] = sign * 1.0 / sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_NOT_FIX_RANDOM_SEED) {
  const size_t n = 3;
  double matrix[n][n] = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true, 1);
  engine.eigenvalue_offset = 6.0;

  double eigvalue;
  vector<double> eigvec(1);  // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);

  auto sign = eigvec[0] / std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] = sign * 1.0 / sqrt(3.0);
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, DYNAMIC_MATRIX) {
  const size_t n = 10;

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for (size_t i = 0; i < n - 1; ++i) {
      out[i] += -1.0 * in[i + 1];
      out[i + 1] += -1.0 * in[i];
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

  LambdaLanczos<double> engine(matmul, n, false, 1);
  engine.init_vector = vector_initializer<double>;
  engine.eps = 1e-14;
  engine.eigenvalue_offset = -10.0;
  double eigvalue;
  vector<double> eigvec(n);
  engine.run(eigvalue, eigvec);

  double correct_eigvalue = -2.0 * cos(M_PI / (n + 1));
  auto sign = eigvec[0] / std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] = sign * std::sin((i + 1) * M_PI / (n + 1));
  }
  lambda_lanczos::util::normalize(correct_eigvec);

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_USE_COMPLEX_TYPE) {
  const size_t n = 3;
  complex<double> matrix[n][n] = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n, true, 1);
  engine.init_vector = vector_initializer<complex<double>>;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);

  vector<complex<double>> correct_eigvec(n);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0) * std::arg(eigvec[0]));
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] = 1.0 / std::sqrt(n) * phase_factor;
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue * engine.eps * 10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, SIMPLE_MATRIX_USE_COMPLEX_TYPE_NOT_FIX_RANDOM_SEED) {
  const size_t n = 3;
  complex<double> matrix[n][n] = {{2.0, 1.0, 1.0}, {1.0, 2.0, 1.0}, {1.0, 1.0, 2.0}};
  /* Its eigenvalues are {4, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n, true, 1);
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);

  vector<complex<double>> correct_eigvec(n);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0) * std::arg(eigvec[0]));
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] = 1.0 / std::sqrt(n) * phase_factor;
  }
  double correct_eigvalue = 4.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue * engine.eps * 10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, HERMITIAN_MATRIX) {
  const size_t n = 3;
  const auto I_ = complex<double>(0.0, 1.0);
  complex<double> matrix[n][n] = {{0.0, I_, 1.0}, {-I_, 0.0, I_}, {1.0, -I_, 0.0}};
  /* Its eigenvalues are {-2, 1, 1} */

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n, false, 1);
  engine.init_vector = vector_initializer<complex<double>>;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);

  vector<complex<double>> correct_eigvec{1.0, I_, -1.0};
  lambda_lanczos::util::normalize(correct_eigvec);
  complex<double> phase_factor = std::exp(complex<double>(0.0, 1.0) * std::arg(eigvec[0]));
  for (size_t i = 0; i < n; ++i) {
    correct_eigvec[i] *= phase_factor;
  }

  double correct_eigvalue = -2.0;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i].real(), eigvec[i].real(), std::abs(correct_eigvalue * engine.eps * 10));
    EXPECT_NEAR(correct_eigvec[i].imag(), eigvec[i].imag(), std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, SINGLE_ELEMENT_MATRIX) {
  const size_t n = 1;

  double correct_eigvalue = 2.0;
  double matrix[n][n] = {{correct_eigvalue}};

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true, 1);
  engine.init_vector = vector_initializer<double>;

  double eigvalue;
  vector<double> eigvec(1);  // The size will be enlarged automatically
  engine.run(eigvalue, eigvec);

  auto sign = eigvec[0] / std::abs(eigvec[0]);
  vector<double> correct_eigvec(n);
  correct_eigvec[0] = sign;

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i], eigvec[i], std::abs(correct_eigvalue * engine.eps * 10));
  }
}

TEST(DIAGONALIZE_TEST, MULTIPLE_EIGENPAIRS) {
  const int n = 8;
  const size_t nroot = 3;

  double matrix[n][n] = {{6, -3, -3, 0, -1, 1, -1, 1},
                         {-3, -4, 2, 2, -1, -5, 0, -4},
                         {-3, 2, 2, -3, 0, 0, -1, -1},
                         {0, 2, -3, 0, -3, 3, 2, 2},
                         {-1, -1, 0, -3, -2, 0, -5, -4},
                         {1, -5, 0, 3, 0, -4, 5, 0},
                         {-1, 0, -1, 2, -5, 5, -4, 4},
                         {1, -4, -1, 2, -4, 0, 4, 2}};

  // the matrix-vector multiplication routine
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<double> engine(mv_mul, n, false, 1);  // false means to calculate the smallest eigenvalue.
  engine.num_eigs = nroot;
  engine.eps = 1e-7;

  vector<double> eigenvalues;
  vector<vector<double>> eigenvectors;
  engine.run(eigenvalues, eigenvectors);

  std::array<double, n> correct_eigvals = {-13.21508597, -8.50033154, -4.26674892};
  std::array<std::array<double, n>, nroot> correct_eigvecs = {
      {{0.02081752, -0.49222707, 0.13202088, 0.24048092, 0.15089223, -0.60850056, 0.48079787, -0.24043829},
       {0.16645991, 0.51818471, -0.00646562, -0.09493495, 0.60595718, 0.02042567, 0.52346924, 0.23043415},
       {0.03381669, -0.07999997, 0.32090331, 0.61650970, 0.41812886, -0.01782613, -0.45571810, 0.35575946}}};

  for (size_t iroot = 0; iroot < nroot; ++iroot) {
    EXPECT_NEAR(correct_eigvals[iroot], eigenvalues[iroot], std::abs(correct_eigvals[iroot] * engine.eps));

    auto sign = eigenvectors[iroot][0] / std::abs(eigenvectors[iroot][0]);
    for (size_t i = 0; i < n; ++i) {
      correct_eigvecs[iroot][i] *= sign;
      EXPECT_NEAR(
          correct_eigvecs[iroot][i], eigenvectors[iroot][i], std::abs(correct_eigvals[iroot] * engine.eps * 10));
    }
  }
}

TEST(DIAGONALIZE_TEST, MULTIPLE_DEGENERATE_EIGENPAIRS) {
  const size_t n = 50;

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for (size_t i = 0; i < n - 1; ++i) {
      out[i] += -1.0 * in[i + 1];
      out[i + 1] += -1.0 * in[i];
    }

    out[0] += -1.0 * in[n - 1];
    out[n - 1] += -1.0 * in[0];
  };
  /*
    This lambda is equivalent to applying following n by n matrix

      0  -1   0       ..     -1
     -1   0  -1       ..      0
      0  -1   0       ..      0
      0     ..        ..      0
      0     ..        0  -1   0
      0     ..       -1   0  -1
     -1     ..        0  -1   0

      Its eigenvalues are -2*cos(2*pi*i/n), 0 <= i < n.
   */

  const int num_eigs = 26;
  LambdaLanczos<double> engine(matmul, n, false, 1);
  engine.num_eigs = num_eigs;
  engine.eps = 1e-14;
  vector<double> eigvals;
  vector<std::vector<double>> eigvecs;
  engine.run(eigvals, eigvecs);

  vector<double> correct_eigvals(engine.num_eigs);
  std::iota(correct_eigvals.begin(), correct_eigvals.end(), -num_eigs / 2);

  std::transform(correct_eigvals.begin(), correct_eigvals.end(), correct_eigvals.begin(), [](double x) {
    return -2.0 * cos(2.0 * M_PI * x / n);
  });
  std::sort(correct_eigvals.begin(), correct_eigvals.end());

  EXPECT_EQ(correct_eigvals.size(), eigvals.size());
  for (size_t i = 0; i < correct_eigvals.size(); ++i) {
    EXPECT_NEAR(correct_eigvals[i], eigvals[i], engine.eps);
  }
}

template <typename T, typename RE>
void generate_random_symmetric_matrix(T** a, vector<T>& eigvec, T& eigvalue, size_t n, size_t rand_n, RE eng) {
  const T min_eigvalue = 1.0;
  std::uniform_int_distribution<size_t> dist_index(0, n - 1);
  std::uniform_real_distribution<double> dist_angle(0.0, 2 * M_PI);
  std::uniform_real_distribution<double> dist_element(min_eigvalue, n * 10);

  /* Generate a random diagonal matrix */
  std::fill(a[0], a[0] + n * n, 0.0);
  T max_eigvalue = min_eigvalue;
  size_t max_eig_index = 0;
  for (size_t i = 0; i < n; ++i) {
    a[i][i] = dist_element(eng);
    if (a[i][i] > max_eigvalue) {
      max_eigvalue = a[i][i];
      max_eig_index = i;
    }
  }

  eigvalue = max_eigvalue;

  /* Eigenvector corresponding to the maximum eigenvalue */
  std::fill(eigvec.begin(), eigvec.end(), T());
  eigvec[max_eig_index] = 1.0;

  for (size_t i = 0; i < rand_n; ++i) {
    size_t k = dist_index(eng);
    size_t l = dist_index(eng);
    while (k == l) {
      l = dist_index(eng);
    }

    T theta = dist_angle(eng);

    T c = std::cos(theta);
    T s = std::sin(theta);
    T a_kk = a[k][k];
    T a_kl = a[k][l];
    T a_ll = a[l][l];

    for (size_t i = 0; i < n; ++i) {
      T aki_next = c * a[k][i] - s * a[l][i];
      a[l][i] = s * a[k][i] + c * a[l][i];
      a[k][i] = aki_next;
    }

    /* Symmetrize */
    for (size_t i = 0; i < n; ++i) {
      a[i][k] = a[k][i];
      a[i][l] = a[l][i];
    }

    a[k][k] = c * (c * a_kk - s * a_kl) - s * (c * a_kl - s * a_ll);
    a[k][l] = s * (c * a_kk - s * a_kl) + c * (c * a_kl - s * a_ll);
    a[l][k] = a[k][l];
    a[l][l] = s * (s * a_kk + c * a_kl) + c * (s * a_kl + c * a_ll);

    T vk_next = c * eigvec[k] - s * eigvec[l];
    eigvec[l] = s * eigvec[k] + c * eigvec[l];
    eigvec[k] = vk_next;
  }
}

TEST(DIAGONALIZE_TEST, RANDOM_SYMMETRIC_MATRIX) {
  const size_t n = 50;

  double** matrix = new double*[n];
  matrix[0] = new double[n * n];
  for (size_t i = 0; i < n; ++i) {
    matrix[i] = matrix[0] + n * i;
  }

  vector<double> correct_eigvec(n);
  double correct_eigvalue = 0.0;

  generate_random_symmetric_matrix(matrix, correct_eigvec, correct_eigvalue, n, n * 10, std::mt19937(1));

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<double> engine(matmul, n, true, 1);
  engine.init_vector = vector_initializer<double>;
  double eigvalue;
  vector<double> eigvec(n);
  engine.run(eigvalue, eigvec);

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));
  auto sign = (eigvec[0] * correct_eigvec[0] > 0) ? 1 : -1;
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvec[i] * sign, eigvec[i], std::abs(correct_eigvalue * engine.eps * n * n));
  }

  delete[] matrix[0];
  delete[] matrix;
}

template <typename T, typename RE>
void generate_random_hermitian_matrix(
    complex<T>** a, vector<complex<T>>& eigvec, T& eigvalue, size_t n, size_t rand_n, RE eng) {
  const complex<T> I_(0, 1);

  const T min_eigvalue = 1.0;
  std::uniform_int_distribution<size_t> dist_index(0, n - 1);
  std::uniform_real_distribution<T> dist_angle(0.0, 2 * M_PI);
  std::uniform_real_distribution<T> dist_element(min_eigvalue, n * 10);

  /* Generate a random diagonal matrix */
  std::fill(a[0], a[0] + n * n, 0.0);
  T max_eigvalue = min_eigvalue;
  size_t max_eig_index = 0;
  for (size_t i = 0; i < n; ++i) {
    T eigvalue_tmp = dist_element(eng);
    a[i][i] = eigvalue_tmp;
    if (eigvalue_tmp > max_eigvalue) {
      max_eigvalue = eigvalue_tmp;
      max_eig_index = i;
    }
  }

  eigvalue = max_eigvalue;

  /* Eigenvector corresponding to the maximum eigenvalue */
  std::fill(eigvec.begin(), eigvec.end(), T());
  eigvec[max_eig_index] = 1.0;

  for (size_t i = 0; i < rand_n; ++i) {
    size_t k = dist_index(eng);
    size_t l = dist_index(eng);
    while (k == l) {
      l = dist_index(eng);
    }

    auto theta = dist_angle(eng);
    auto phi1 = dist_angle(eng);
    auto phi2 = dist_angle(eng);

    complex<T> u_kk = std::exp(I_ * phi1) * std::cos(theta);
    complex<T> u_kl = -std::exp(I_ * phi2) * std::sin(theta);
    complex<T> u_lk = std::exp(-I_ * phi2) * std::sin(theta);
    complex<T> u_ll = std::exp(-I_ * phi1) * std::cos(theta);

    auto a_kk = a[k][k];
    auto a_kl = a[k][l];
    auto a_lk = a[l][k];
    auto a_ll = a[l][l];

    for (size_t i = 0; i < n; ++i) {
      auto aki_next = u_kk * a[k][i] + u_kl * a[l][i];
      a[l][i] = u_lk * a[k][i] + u_ll * a[l][i];
      a[k][i] = aki_next;
    }

    /* Hermitize */
    for (size_t i = 0; i < n; ++i) {
      a[i][k] = std::conj(a[k][i]);
      a[i][l] = std::conj(a[l][i]);
    }

    a[k][k] = u_kk * (a_kk * std::conj(u_kk) + a_kl * std::conj(u_kl)) +
              u_kl * (a_lk * std::conj(u_kk) + a_ll * std::conj(u_kl));
    a[k][l] = u_kk * (a_kk * std::conj(u_lk) + a_kl * std::conj(u_ll)) +
              u_kl * (a_lk * std::conj(u_lk) + a_ll * std::conj(u_ll));
    a[l][k] = std::conj(a[k][l]);
    a[l][l] = u_lk * (a_kk * std::conj(u_lk) + a_kl * std::conj(u_ll)) +
              u_ll * (a_lk * std::conj(u_lk) + a_ll * std::conj(u_ll));

    auto vk_next = u_kk * eigvec[k] + u_kl * eigvec[l];
    eigvec[l] = u_lk * eigvec[k] + u_ll * eigvec[l];
    eigvec[k] = vk_next;
  }
}

TEST(DIAGONALIZE_TEST, RANDOM_HERMITIAN_MATRIX) {
  const size_t n = 10;

  complex<double>** matrix = new complex<double>*[n];
  matrix[0] = new complex<double>[n * n];
  for (size_t i = 0; i < n; ++i) {
    matrix[i] = matrix[0] + n * i;
  }

  vector<complex<double>> correct_eigvec(n);
  double correct_eigvalue = 0.0;

  generate_random_hermitian_matrix(matrix, correct_eigvec, correct_eigvalue, n, n * 10, std::mt19937(1));

  auto matmul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j] * in[j];
      }
    }
  };

  LambdaLanczos<complex<double>> engine(matmul, n, true, 1);
  engine.init_vector = vector_initializer<complex<double>>;
  engine.eps = 1e-14;
  double eigvalue;
  vector<complex<double>> eigvec(n);
  engine.run(eigvalue, eigvec);

  EXPECT_NEAR(correct_eigvalue, eigvalue, std::abs(correct_eigvalue * engine.eps));

  const complex<double> I_(0, 1);
  auto phase = std::exp(I_ * (std::arg(eigvec[0]) - std::arg(correct_eigvec[0])));
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR((correct_eigvec[i] * phase).real(), eigvec[i].real(), std::abs(correct_eigvalue * engine.eps * 10));
    EXPECT_NEAR((correct_eigvec[i] * phase).imag(), eigvec[i].imag(), std::abs(correct_eigvalue * engine.eps * 10));
  }

  delete[] matrix[0];
  delete[] matrix;
}

TEST(TRIDIAGONAL_TEST, IMPLICIT_SHIFT_QR) {
  vector<double> alpha{1, 2, 3};
  vector<double> beta{2, 2};

  const auto n = alpha.size();
  vector<double> eigvals(n);
  vector<vector<double>> eigvecs;

  lambda_lanczos::tridiagonal_impl::tridiagonal_eigenpairs(alpha, beta, eigvals, eigvecs);

  vector<double> correct_eigvals{-1, 2, 5};
  vector<vector<double>> correct_eigvecs{{2, -2, 1}, {2, 1, -2}, {1, 2, 2}};
  for (auto& v : correct_eigvecs) {
    lambda_lanczos::util::normalize(v);
  }

  double eps = 1e-10;
  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(correct_eigvals[i], eigvals[i], eps);

    auto sign = eigvecs[i][0] / std::abs(eigvecs[i][0]);
    std::cout << lambda_lanczos::util::vectorToString(eigvecs[i]) << std::endl;
    for (size_t j = 0; j < n; ++j) {
      EXPECT_NEAR(correct_eigvecs[i][j] * sign, eigvecs[i][j], eps);
    }
  }
  std::cout << std::endl;
}

TEST(TRIDIAGONAL_TEST, NULL_EIGENVALUE_NO_ASSERTS) {
  vector<double> alpha{6.82333617e-03, 3.09398208e+00, 1.89919458e+00, 1.28531906e-16};
  vector<double> beta{1.19582528e-01, -1.37689656e+00, 6.16147405e-15};

  vector<double> v = alpha;
  const auto n = alpha.size();

  vector<vector<double>> q;

  lambda_lanczos::tridiagonal_impl::tridiagonal_eigenpairs(alpha, beta, v, q);

  for (size_t i = 0; i < n; ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}
