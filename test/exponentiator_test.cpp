#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif

#include <taylor_exponentiator.hpp>
#include <exponentiator.hpp>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <gtest/gtest.h>



template <typename MT, typename T>
std::vector<T> apply_matrix(const MT& matrix,
                            const std::vector<T>& in,
                            bool dagger = false) {
  const auto n = in.size();
  std::vector<T> out(n, 0.0);

  for(size_t i = 0; i < n; ++i) {
    for(size_t j = 0; j < n; ++j) {
      if(dagger) {
        out[i] += lambda_lanczos::util::typed_conj(matrix[j][i]) * in[j];
      } else {
        out[i] += matrix[i][j] * in[j];
      }
    }
  }

  return out;
}


TEST(EXPONENTIATOR_TEST, EXPONENTIATE_REAL) {
  using namespace std;
  using namespace lambda_lanczos;

  const size_t n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
                          {1.0, 1.0, 2.0} };

  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(size_t i = 0; i < n; ++i) {
      for(size_t j = 0; j < n; ++j) {
        out[i] += matrix[i][j]*in[j];
      }
    }
  };

  double a = 3;
  lambda_lanczos::Exponentiator<double> exponentiator(mv_mul, n);
  vector<double> input = {1, 0, 0};
  vector<double> output(n);
  size_t itern = exponentiator.run(a, input, output);


  double u[n][n] = { {1/sqrt(3), -1/sqrt(2), -1/sqrt(6)},
                     {1/sqrt(3),  0/sqrt(2),  2/sqrt(6)},
                     {1/sqrt(3),  1/sqrt(2), -1/sqrt(6)} }; // u[:][k] is the k-th eigenvector
  vector<double> eigvals = {4, 1, 1};
  vector<vector<double>> diag(n, vector<double>(n, 0.0));
  for(size_t i = 0; i < n; ++i) {
    diag[i][i] = exp(a * eigvals[i]);
  }

  auto tmp = input;
  tmp = apply_matrix(u, tmp, true);
  tmp = apply_matrix(diag, tmp);
  tmp = apply_matrix(u, tmp);

  double overlap = std::abs(util::inner_prod(tmp, output))/util::norm(tmp)/util::norm(output);

  cout << setprecision(16);
  cout << "itern: " << itern << endl;
  cout << "overlap: " << overlap << endl;

  EXPECT_NEAR(1.0, overlap, exponentiator.eps);


  /* Check Taylor exponentiation */
  itern = exponentiator.taylor_run(a, input, output);
  overlap = std::abs(util::inner_prod(tmp, output))/util::norm(tmp)/util::norm(output);

  cout << "itern (Taylor): " << itern << endl;
  cout << "overlap (Taylor): " << overlap << endl;
  EXPECT_NEAR(1.0, overlap, exponentiator.eps);
}


void make_plane_wave(double t,
                     size_t n,
                     std::vector<double>& ev,
                     std::vector<std::vector<std::complex<double>>>& u) {
  using namespace std;

  const static auto I_ = complex<double>(0.0, 1.0);

  vector<pair<double, double>> ke;
  for(size_t j = 0; j < n; ++j) {
    double k = 2*M_PI/n*j;
    ke.emplace_back(k, 2*t*cos(k));
  }

  sort(ke.begin(), ke.end(),
       [](const auto& x, const auto& y) {
         return x.second < y.second;
       });

  ev = vector<double>(n);
  u = vector<vector<complex<double>>>(n, vector<complex<double>>(n, 0));
  for(size_t j = 0; j < n; ++j) {
    ev[j] = ke[j].second;
    for(size_t i = 0; i < n; ++i) {
      u[i][j] = exp(I_ * ke[j].first * (double)i)/sqrt(n);
    }
  }
}


TEST(EXPONENTIATOR_TEST, EXPONENTIATE_LARGE_MATRIX) {
  using namespace std;
  using namespace lambda_lanczos;

  const size_t n = 100;
  const double t = -1.0;

  auto mv_mul = [&](const vector<complex<double>>& in, vector<complex<double>>& out) {
    for(size_t i = 0; i < n-1; ++i) {
      out[i] += t*in[i+1];
      out[i+1] += t*in[i];
    }

    out[0] += t*in[n-1];
    out[n-1] += t*in[0];
  };

  complex<double> a(0.0, 3.0);

  lambda_lanczos::Exponentiator<complex<double>> exponentiator(mv_mul, n);
  vector<complex<double>> input(n);
  input[0] = complex<double>(1, 2);
  input[n-1] = complex<double>(1, 2);
  input[n/2] = complex<double>(8, 2);
  util::normalize(input);
  vector<complex<double>> output(n);
  size_t itern = exponentiator.run(a, input, output);

  vector<double> eigvals;
  vector<vector<complex<double>>> u;
  make_plane_wave(t, n, eigvals, u);
  vector<vector<complex<double>>> diag(n, vector<complex<double>>(n, 0.0));
  for(size_t i = 0; i < n; ++i) {
    diag[i][i] = exp(a * eigvals[i]);
  }

  auto tmp = input;
  tmp = apply_matrix(u, tmp, true);
  tmp = apply_matrix(diag, tmp);
  tmp = apply_matrix(u, tmp);

  double overlap = std::abs(util::inner_prod(tmp, output))/util::norm(tmp)/util::norm(output);

  cout << setprecision(16);
  cout << "itern: " << itern << endl;
  cout << "overlap: " << overlap << endl;

  EXPECT_NEAR(1.0, overlap, exponentiator.eps);



  /* Check Taylor exponentiation */
  itern = exponentiator.taylor_run(a, input, output);
  overlap = std::abs(util::inner_prod(tmp, output))/util::norm(tmp)/util::norm(output);

  cout << "itern (Taylor): " << itern << endl;
  cout << "overlap (Taylor): " << overlap << endl;
  EXPECT_NEAR(1.0, overlap, exponentiator.eps);
}
