#ifndef LAMBDA_LANCZOS_H_
#define LAMBDA_LANCZOS_H_

#include <iostream>
#include <vector>
#include <tuple>
#include <functional>
#include <cassert>
#include <limits>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <utility>
#include "lambda_lanczos_util.hpp"
#include "lambda_lanczos_tridiagonal.hpp"


namespace lambda_lanczos {
/**
 * @brief Computes an eigenvector corresponding to given eigenvalue for the original matrix.
 */
template <typename T, typename LT>
inline auto eigenvector(const std::vector<T>& alpha,
                        const std::vector<T>& beta,
                        const std::vector<std::vector<LT>> u,
                        T ev) -> std::vector<decltype(T()+LT())> {
  const auto m = alpha.size();
  const auto n = u[0].size();

  std::vector<LT> eigvec(n, 0.0);

  auto cv = tridiagonal::tridiagonal_eigenvector(alpha, beta, ev);

  for (size_t k = m; k-- > 0;) {
    for (size_t i = 0; i < n; ++i) {
      eigvec[i] += cv[k] * u[k][i];
    }
  }

  util::normalize(eigvec);

  return eigvec;
}



/**
 * @brief Template class to implement random vector initializer.
 *
 * "Partially specialization of function" is not allowed,
 * so here it is mimicked by wrapping the "init" function with a class template.
 */
template <typename T>
struct VectorRandomInitializer {
public:
  /**
   * @brief Initialize given vector randomly in the range of [-1, 1].
   *
   * For complex type, the real and imaginary part of each element will be initialized in
   * the range of [-1, 1].
   */
  static void init(std::vector<T>& v) {
    std::random_device dev;
    std::mt19937 mt(dev());
    std::uniform_real_distribution<T> rand((T)(-1.0), (T)(1.0));

    size_t n = v.size();
    for(size_t i = 0; i < n; ++i) {
      v[i] = rand(mt);
    }
  }
};


template <typename T>
struct VectorRandomInitializer<std::complex<T>> {
public:
  static void init(std::vector<std::complex<T>>& v) {
    std::random_device dev;
    std::mt19937 mt(dev());
    std::uniform_real_distribution<T> rand((T)(-1.0), (T)(1.0));

    size_t n = v.size();
    for(size_t i = 0; i < n; ++i) {
      v[i] = std::complex<T>(rand(mt), rand(mt));
    }
  }
};


/**
 * @brief Calculation engine for Lanczos algorithm
 */
template <typename T>
class LambdaLanczos {
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
   * into `out`, where `A` is the matrix to be diagonalized.
   * You can assume the output vector `out` has been initialized with zeros before the `mv_mul` is called.
   */
  std::function<void(const std::vector<T>& in, std::vector<T>& out)> mv_mul;

  /** @brief Function to initialize the initial Lanczos vector.
   *
   * After this function called, the output vector will be normalized automatically.
   * Default value is #lambda_lanczos::VectorRandomInitializer::init.
   */
  std::function<void(std::vector<T>& vec)> init_vector = VectorRandomInitializer<T>::init;

  /** @brief Dimension of the matrix to be diagonalized. */
  size_t matrix_size;
  /** @brief Iteration limit of Lanczos algorithm, set to `matrix_size` automatically. */
  size_t max_iteration;
  /** @brief Convergence threshold of Lanczos iteration.
   *
   * `eps` = 1e-12 means that the eigenvalue will be calculated with 12 digits of precision.
   *
   * Default value is system-dependent. On usual 64-bit systems:
   * | type (including complex one)       | size (system-dep.) | `eps`   |
   * | ---------------------------------- | ------------------ | ------- |
   * | float                              | 4 bytes            | 1e-4    |
   * | double                             | 8 bytes            | 1e-12   |
   * | long double                        | 16 bytes           | 1e-19   |
   */
  real_t<T> eps = util::minimum_effective_decimal<real_t<T>>() * 1e3;

  /** @brief true to calculate maximum eigenvalue, false to calculate minimum one.*/
  bool find_maximum;

  /**
   * @brief Shifts the eigenvalues of the given matrix A.
   *
   * The algorithm will calculate the eigenvalue of matrix (A+`eigenvalue_offset`*E),
   * here E is the identity matrix. The result eigenvalue from `run()` will take this shifting into account,
   * so you don't have to "reshift" the result with `eigenvalue_offset`.
   **/
  real_t<T> eigenvalue_offset = 0.0;

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
    * @brief Constructs Lanczos calculation engine.
    *
    * @param mv_mul Matrix-vector multiplication routine. See #mv_mul for details.
    * @param matrix_size The size of your matrix, i.e. if your matrix is n by n,
    * `matrix_size` should be n.
    * @param find_maximum specifies which of the minimum or maximum eigenvalue to be calculated.
    * By default, `find_maximum=false` so the library will calculates the minimum one.
    */
  LambdaLanczos(std::function<void(const std::vector<T>&, std::vector<T>&)> mv_mul, size_t matrix_size, bool find_maximum = false) :
    mv_mul(mv_mul), matrix_size(matrix_size), max_iteration(matrix_size), find_maximum(find_maximum) {}

  /**
   * @brief Executes Lanczos algorithm and stores the result into reference variables passed as arguments.
   * @return Lanczos-iteration count
   */
  size_t run(std::vector<real_t<T>>& eigvalues, std::vector<std::vector<T>>& eigvecs) const {
    const size_t nroot = eigvecs.size();
    assert(eigvalues.size()==nroot);
    assert(0 < this->tridiag_eps_ratio && this->tridiag_eps_ratio < 1);

    std::vector<std::vector<T>> u; // Lanczos vectors
    std::vector<real_t<T>> alpha;  // Diagonal elements of an approximated tridiagonal matrix
    std::vector<real_t<T>> beta;   // Subdiagonal elements of an approximated tridiagonal matrix

    u.reserve(this->initial_vector_size);
    alpha.reserve(this->initial_vector_size);
    beta.reserve(this->initial_vector_size);

    const auto n = this->matrix_size;

    u.emplace_back(n);
    this->init_vector(u[0]);
    util::normalize(u[0]);

    std::vector<real_t<T>> evs(nroot); // Calculated eigenvalue
    std::vector<real_t<T>> pevs(nroot, std::numeric_limits<real_t<T>>::max()); // Previous eigenvalue

    size_t itern = this->max_iteration;
    for(size_t k = 1; k <= this->max_iteration; ++k) {
      /* au = (A + offset*E)uk, here E is the identity matrix */
      std::vector<T> au(n, 0.0); // Temporal storage to store matrix-vector multiplication result
      this->mv_mul(u[k-1], au);
      for(size_t i = 0; i < n; ++i) {
        au[i] += u[k-1][i]*this->eigenvalue_offset;
      }

      alpha.push_back(std::real(util::inner_prod(u[k-1], au)));

      u.push_back(std::move(au));
      for(size_t i = 0; i < n; ++i) {
        if(k == 1) {
          u[k][i] = u[k][i] - alpha[k-1]*u[k-1][i];
        } else {
          u[k][i] = u[k][i] - beta[k-2]*u[k-2][i] - alpha[k-1]*u[k-1][i];
        }
      }

      util::schmidt_orth(u[k], u.begin(), u.end()-1);

      beta.push_back(util::norm(u[k]));

      for (size_t iroot = 0; iroot < nroot; ++iroot) {
        evs[iroot] =
          tridiagonal::find_mth_eigenvalue(alpha, beta,
                                           this->find_maximum ? alpha.size()-1-iroot : iroot,
                                           this->eps * this->tridiag_eps_ratio);
      }

      const real_t<T> zero_threshold = util::minimum_effective_decimal<real_t<T>>()*1e-1;
      if(beta.back() < zero_threshold) {
        itern = k;
        break;
      }

      util::normalize(u[k]);

      /*
       * only break loop if convergence condition is met for all roots
       */
      bool break_cond = true;
      for(size_t iroot = 0; iroot < nroot; ++iroot) {
        const auto& ev = evs[iroot];
        const auto& pev = pevs[iroot];
        if (std::abs(ev - pev) >= std::min(std::abs(ev), std::abs(pev)) * this->eps){
          break_cond = false;
          break;
        }
      }

      if (break_cond) {
        itern = k;
        break;
      } else {
        pevs = evs;
      }
    }

    eigvalues = evs;
    beta.back() = 0.0;

    for(size_t iroot = 0; iroot < nroot; ++iroot) {
      auto& eigvec = eigvecs[iroot];
      auto& ev = eigvalues[iroot];

      eigvec = eigenvector(alpha, beta, u, ev);
      ev -= this->eigenvalue_offset;
    }

    return itern;
  }

  /**
   * @brief Executes Lanczos algorithm and return result as a tuple.
   *
   * This function provides C++17 multiple-value-return interface.
   *
   * @return Eigenvalue
   * @return Eigenvector
   * @return Lanczos-iteration count
   */
  std::tuple<real_t<T>, std::vector<T>, size_t> run() const {
    real_t<T> eigvalue;
    std::vector<T> eigvec(this->matrix_size);
    size_t itern = this->run(eigvalue, eigvec);

    return {eigvalue, eigvec, itern};
  }

  size_t run(real_t<T>& eigvalue, std::vector<T>& eigvec) const{
    std::vector<real_t<T>> eigvalues(1);
    std::vector<std::vector<T>> eigvecs(1);
    auto retval = run(eigvalues, eigvecs);
    eigvalue = eigvalues[0];
    eigvec = std::move(eigvecs[0]);
    return retval;
  }
};


} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_H_ */
