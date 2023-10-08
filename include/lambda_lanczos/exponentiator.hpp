#ifndef LAMBDA_LANCZOS_EXPONENTIATOR_H_
#define LAMBDA_LANCZOS_EXPONENTIATOR_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "lambda_lanczos.hpp"
#include "lambda_lanczos_util.hpp"

namespace lambda_lanczos {

/**
 * @brief Calculation engine for Lanczos exponentiation
 */
template <typename T>
class Exponentiator {
 private:
  /**
   * @brief See #util::real_t for details.
   */
  template <typename n_type>
  using real_t = util::real_t<n_type>;

 public:
  /**
   * @brief Matrix-vector multiplication routine.
   *
   * This must be a function to calculate `A*in` and store the result
   * into `out`, where `A` is the matrix to be exponentiated.
   * You can assume the output vector `out` has been initialized with zeros before the `mv_mul` is called.
   */
  std::function<void(const std::vector<T>& in, std::vector<T>& out)> mv_mul;

  /** @brief Dimension of the matrix to be exponentiated. */
  size_t matrix_size;
  /** @brief Iteration limit of Lanczos algorithm, set to `matrix_size` automatically. */
  size_t max_iteration;
  /** @brief Convergence threshold of Lanczos iteration.
   *
   * `eps` = 1e-14 means that the eigenvalue will be calculated with 14 digits of precision.
   *
   * Default value is system-dependent. On usual 64-bit systems:
   * | type (including complex one)       | size (system-dep.) | `eps`   |
   * | ---------------------------------- | ------------------ | ------- |
   * | float                              | 4 bytes            | 1e-6    |
   * | double                             | 8 bytes            | 1e-14   |
   * | long double                        | 16 bytes           | 1e-21   |
   */
  real_t<T> eps = std::numeric_limits<real_t<T>>::epsilon() * 1e2;

  /**
   * @brief Flag to execute explicit Lanczos-vector orthogonalization.
   */
  bool full_orthogonalize = false;

  /** @brief (Not necessary to change)
   *
   * This variable specifies the initial reserved size of Lanczos vectors.
   * Controlling this variable might affect reserving efficiency,
   * but that would be very small compared to matrix-vector-multiplication cost.
   */
  size_t initial_vector_size = 200;

  /**
   * @brief Constructs Lanczos exponetiation engine.
   *
   * @param mv_mul Matrix-vector multiplication routine. See #mv_mul for details.
   * @param matrix_size The size of your matrix, i.e. if your matrix is n by n,
   * `matrix_size` should be n.
   */
  Exponentiator(std::function<void(const std::vector<T>&, std::vector<T>&)> mv_mul, size_t matrix_size)
      : mv_mul(mv_mul), matrix_size(matrix_size), max_iteration(matrix_size) {}

  /**
   * @brief Apply matrix exponentiation exp(a*A) to `input` and store the result into `output`.
   * @return Lanczos-iteration count
   */
  size_t run(const T& a, const std::vector<T>& input, std::vector<T>& output) const {
    assert(input.size() == this->matrix_size);

    std::vector<std::vector<T>> u;  // Lanczos vectors
    std::vector<real_t<T>> alpha;   // Diagonal elements of an approximated tridiagonal matrix
    std::vector<real_t<T>> beta;    // Subdiagonal elements of an approximated tridiagonal matrix

    u.reserve(this->initial_vector_size);
    alpha.reserve(this->initial_vector_size);
    beta.reserve(this->initial_vector_size);

    const auto n = this->matrix_size;

    u.push_back(input);
    util::normalize(u[0]);

    std::vector<T> coeff_prev;

    size_t itern = this->max_iteration;
    for (size_t k = 1; k <= this->max_iteration; ++k) {
      u.emplace_back(n, 0.0);
      this->mv_mul(u[k - 1], u[k]);

      alpha.push_back(std::real(util::inner_prod(u[k - 1], u[k])));

      for (size_t i = 0; i < n; ++i) {
        if (k == 1) {
          u[k][i] = u[k][i] - alpha[k - 1] * u[k - 1][i];
        } else {
          u[k][i] = u[k][i] - beta[k - 2] * u[k - 2][i] - alpha[k - 1] * u[k - 1][i];
        }
      }

      if (this->full_orthogonalize) {
        util::schmidt_orth(u[k], u.begin(), u.end() - 1);
      }

      std::vector<real_t<T>> ev(alpha.size());
      std::vector<std::vector<real_t<T>>> p(alpha.size());
      lambda_lanczos::tridiagonal::tridiagonal_eigenpairs(alpha, beta, ev, p);

      std::vector<T> coeff(alpha.size(), 0.0);
      for (size_t i = 0; i < alpha.size(); ++i) {
        for (size_t j = 0; j < alpha.size(); ++j) {
          coeff[i] += p[j][i] * std::exp(a * ev[j]) * p[j][0];
        }
      }

      /*std::cout << "beta:" << std::endl;
      for(size_t i = 0; i < beta.size(); ++i) {
        std::cout << i << " : " << beta[i] << std::endl;
      }

      std::cout << "coeff:" << std::endl;
      for(size_t i = 0; i < coeff.size(); ++i) {
        std::cout << i << " : " << coeff[i] << std::endl;
      }*/

      beta.push_back(util::norm(u[k]));

      T overlap = 0.0;
      for (size_t i = 0; i < coeff_prev.size(); ++i) {
        overlap += util::typed_conj(coeff_prev[i]) * coeff[i];  // Last element of coeff is not used here
      }

      coeff_prev = std::move(coeff);

      const real_t<T> beta_threshold = std::numeric_limits<real_t<T>>::epsilon();
      if (std::abs(1 - std::abs(overlap)) < eps || beta.back() < beta_threshold) {
        itern = k;
        break;
      }

      util::normalize(u[k]);
    }

    output.resize(n);
    std::fill(output.begin(), output.end(), T());
    const T norm = util::norm(input);
    for (size_t l = 0; l < coeff_prev.size(); ++l) {
      for (size_t i = 0; i < n; ++i) {
        output[i] += norm * coeff_prev[l] * u[l][i];
      }
    }

    return itern;
  }

  size_t taylor_run(const T& a, const std::vector<T>& input, std::vector<T>& output) {
    const size_t n = this->matrix_size;
    assert(this->matrix_size == input.size());

    if (a == T()) {  // Zero check
      output = input;
      return 1;
    }

    std::vector<std::vector<T>> taylors;
    taylors.push_back(input);

    T factor = 1.0;
    for (size_t k = 1;; ++k) {
      factor *= a / (T)k;
      taylors.emplace_back(n, 0.0);
      mv_mul(taylors[k - 1], taylors[k]);

      if (lambda_lanczos::util::norm(taylors[k]) * std::abs(factor) < eps) {
        break;
      }
    }

    /* Sum Taylor series backward */
    output.resize(n);
    std::fill(output.begin(), output.end(), 0.0);
    for (size_t k = taylors.size(); k-- > 0;) {
      for (size_t i = 0; i < n; ++i) {
        output[i] += taylors[k][i] * factor;
      }

      factor *= (T)k / a;
    }

    return taylors.size();
  }
};

} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_EXPONENTIATOR_H_ */
