#ifndef LAMBDA_LANCZOS_EXPONENTIATOR_H_
#define LAMBDA_LANCZOS_EXPONENTIATOR_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>
#include <functional>
#include <cassert>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
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
  real_t<T> eps = util::minimum_effective_decimal<real_t<T>>() * 1e1;

  /** @brief (Not necessary to change)
   *
   * Description for those who know the Lanczos algorithm:
   * This is the ratio between the convergence threshold of resulted eigenvalue and the that of
   * tridiagonal eigenvalue. To convergent whole Lanczos algorithm,
   * the convergence threshold for the tridiagonal eigenvalue should be smaller than `eps`.
   */
  real_t<T> tridiag_eps_ratio = 1e-1;

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
  Exponentiator(std::function<void(const std::vector<T>&, std::vector<T>&)> mv_mul, size_t matrix_size) :
    mv_mul(mv_mul), matrix_size(matrix_size), max_iteration(matrix_size) {}

  /**
   * @brief Apply matrix exponentiation exp(a*A) to `input` and store the result into `output`.
   * @return Lanczos-iteration count
   */
  size_t run(const T& a,
             const std::vector<T>& input,
             std::vector<T>& output) const {
    assert(0 < this->tridiag_eps_ratio && this->tridiag_eps_ratio < 1);
    assert(input.size() == this->matrix_size);

    std::vector<std::vector<T>> u; // Lanczos vectors
    std::vector<real_t<T>> alpha;  // Diagonal elements of an approximated tridiagonal matrix
    std::vector<real_t<T>> beta;   // Subdiagonal elements of an approximated tridiagonal matrix

    u.reserve(this->initial_vector_size);
    alpha.reserve(this->initial_vector_size);
    beta.reserve(this->initial_vector_size);

    const auto n = this->matrix_size;

    std::vector<T> au(n, 0.0); // Temporal storage to store matrix-vector multiplication result

    u.push_back(input);
    util::normalize(u[0]);

    std::vector<T> coeff;

    size_t itern = this->max_iteration;
    for(size_t k = 1; k <= this->max_iteration; ++k) {
      std::fill(au.begin(), au.end(), 0.0);
      this->mv_mul(u[k-1], au);

      alpha.push_back(std::real(util::inner_prod(u[k-1], au)));

      u.emplace_back(n);
      u[k] = au;
      for(size_t i = 0; i < n; ++i) {
        if(k == 1) {
          u[k][i] = au[i] - alpha[k-1]*u[k-1][i];
        } else {
          u[k][i] = au[i] - beta[k-2]*u[k-2][i] - alpha[k-1]*u[k-1][i];
        }
      }

      util::schmidt_orth(u[k], u.begin(), u.end()-1);

      coeff = std::vector<T>(alpha.size(), 0.0);

      std::vector<real_t<T>> ev(alpha.size());
      std::vector<std::vector<real_t<T>>> p(alpha.size());
      for(size_t j = 0; j < alpha.size(); ++j) {
        ev[j] = lambda_lanczos::tridiagonal::
          find_mth_eigenvalue(alpha, beta, j, this->eps * this->tridiag_eps_ratio);
        p[j] = lambda_lanczos::tridiagonal::
          tridiagonal_eigenvector(alpha, beta, ev[j]);
      }

      for(size_t i = 0; i < alpha.size(); ++i) {
        for(size_t j = 0; j < alpha.size(); ++j) {
          coeff[i] += p[j][i] * std::exp(a*ev[j]) * p[j][0];
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

      const real_t<T> beta_threshold = util::minimum_effective_decimal<real_t<T>>()*1e1;
      if(std::abs(coeff.back()) < eps || beta.back() < beta_threshold) {
        itern = k;
        break;
      }

      util::normalize(u[k]);
    }

    const T norm = util::norm(input);
    output = std::vector<T>(n, 0.0);
    for(size_t k = 0;k < alpha.size(); ++k) {
      for(size_t i = 0;i < n; ++i) {
        output[i] += norm*coeff[k]*u[k][i];
      }
    }

    return itern;
  }
};


} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_EXPONENTIATOR_H_ */
