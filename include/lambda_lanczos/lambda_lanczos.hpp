#ifndef LAMBDA_LANCZOS_H_
#define LAMBDA_LANCZOS_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "eigenpair_manager.hpp"
#include "lambda_lanczos_tridiagonal.hpp"
#include "lambda_lanczos_util.hpp"

namespace lambda_lanczos {
/**
 * @brief Computes the eigenvectors from Krylov subspace information.
 *
 * @param [in] alpha Diagonal elements of the tridiagonal matrix.
 * @param [in] beta Sub-diagonal elements of the tridiagonal matrix.
 * @param [in] u Lanczos vectors.
 * @param [in] find_maximum True to calculate maximum eigenvalues. False to calculate minimum eigenvalues.
 * @param [in] num_of_eigenvalues Number of eigenvalues to be calculated.
 *
 * @return Calculated eigenvectors.
 */
template <typename T, typename LT>
inline auto compute_eigenvectors(const std::vector<T>& alpha,
                                 const std::vector<T>& beta,
                                 const std::vector<std::vector<LT>>& u,
                                 const bool find_maximum,
                                 const size_t num_of_eigenvalues) -> std::vector<std::vector<decltype(T() + LT())>> {
  const auto m = alpha.size();
  const auto n = u[0].size();

  std::vector<T> tridiagonal_eigenvalues;
  std::vector<std::vector<T>> tridiagonal_eigenvectors;

  tridiagonal::tridiagonal_eigenpairs(alpha, beta, tridiagonal_eigenvalues, tridiagonal_eigenvectors);

  std::vector<std::vector<LT>> eigenvectors;
  for (size_t i = 0; i < num_of_eigenvalues; ++i) {
    eigenvectors.emplace_back(n);
  }

  for (size_t index = 0; index < num_of_eigenvalues; index++) {
    size_t index_tri = find_maximum ? m - index - 1 : index;
    for (size_t k = m; k-- > 0;) {
      for (size_t i = 0; i < n; ++i) {
        eigenvectors[index][i] += tridiagonal_eigenvectors[index_tri][k] * u[k][i];
      }
    }
    util::normalize(eigenvectors[index]);
  }

  return eigenvectors;
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
    for (size_t i = 0; i < n; ++i) {
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
    for (size_t i = 0; i < n; ++i) {
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
  real_t<T> eps = std::numeric_limits<real_t<T>>::epsilon() * 1e3;

  /** @brief true to calculate maximum eigenvalue, false to calculate minimum one.*/
  bool find_maximum;

  /** @brief Number of eigenpairs to be calculated. */
  size_t num_eigs = 1;

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
   * This variable specifies the number of eigenpairs to be calculated per Lanczos iteration.
   * For example, when `num_eigs == 20` and `num_eigs_per_iteration == 5`,
   * `run()` will executes 4 Lanczos iterations.
   */
  size_t num_eigs_per_iteration = 5;

  /** @brief (Not necessary to change)
   *
   * This variable specifies the initial reserved size of Lanczos vectors.
   * Controlling this variable might affect reserving efficiency,
   * but that would be very small compared to matrix-vector-multiplication cost.
   */
  size_t initial_vector_size = 200;

 private:
  /**
   * @brief Iteration counts of the latest run.
   */
  std::vector<size_t> iter_counts;

 public:
  /**
   * @brief Constructs Lanczos calculation engine.
   *
   * @param mv_mul Matrix-vector multiplication routine. See #mv_mul for details.
   * @param matrix_size The size of your matrix, i.e. if your matrix is n by n,
   * `matrix_size` should be n.
   * @param find_maximum specifies which of the minimum or maximum eigenvalue to be calculated.
   * @param num_eigs specifies how many eigenpairs to be calculate, e.g.,
   * if `find_maximum = true` and `num_eig = 3`, LambdaLanczos calculates 3 maximum eigenpairs.
   */
  LambdaLanczos(std::function<void(const std::vector<T>&, std::vector<T>&)> mv_mul,
                size_t matrix_size,
                bool find_maximum,
                size_t num_eigs)
      : mv_mul(mv_mul),
        matrix_size(matrix_size),
        max_iteration(matrix_size),
        find_maximum(find_maximum),
        num_eigs(num_eigs) {}

  /**
   * @brief Not documented (In most cases, `run()` is preferred).
   *
   * @details Lanczos algorithm and stores the result into reference variables passed as arguments.
   * @return Lanczos-iteration count
   */
  template <typename Iterable>
  size_t run_iteration(std::vector<real_t<T>>& eigvalues,
                       std::vector<std::vector<T>>& eigvecs,
                       size_t nroot,
                       Iterable orthogonalizeTo) const {
    std::vector<std::vector<T>> u;  // Lanczos vectors
    std::vector<real_t<T>> alpha;   // Diagonal elements of an approximated tridiagonal matrix
    std::vector<real_t<T>> beta;    // Subdiagonal elements of an approximated tridiagonal matrix

    u.reserve(this->initial_vector_size);
    alpha.reserve(this->initial_vector_size);
    beta.reserve(this->initial_vector_size);

    const auto n = this->matrix_size;

    u.emplace_back(n);
    this->init_vector(u[0]);
    util::schmidt_orth(u[0], orthogonalizeTo.cbegin(), orthogonalizeTo.cend());
    util::normalize(u[0]);

    std::vector<real_t<T>> evs;   // Calculated eigenvalue
    std::vector<real_t<T>> pevs;  // Previous eigenvalue

    size_t itern = this->max_iteration;
    for (size_t k = 1; k <= this->max_iteration; ++k) {
      /* au = (A + offset*E)uk, here E is the identity matrix */
      std::vector<T> au(n, 0.0);  // Temporal storage to store matrix-vector multiplication result
      this->mv_mul(u[k - 1], au);
      for (size_t i = 0; i < n; ++i) {
        au[i] += u[k - 1][i] * this->eigenvalue_offset;
      }

      alpha.push_back(std::real(util::inner_prod(u[k - 1], au)));

      u.push_back(std::move(au));
      for (size_t i = 0; i < n; ++i) {
        if (k == 1) {
          u[k][i] = u[k][i] - alpha[k - 1] * u[k - 1][i];
        } else {
          u[k][i] = u[k][i] - beta[k - 2] * u[k - 2][i] - alpha[k - 1] * u[k - 1][i];
        }
      }

      util::schmidt_orth(u[k], orthogonalizeTo.cbegin(), orthogonalizeTo.cend());
      util::schmidt_orth(u[k], u.begin(), u.end() - 1);

      beta.push_back(util::norm(u[k]));

      size_t num_eigs_to_calculate = std::min(nroot, alpha.size());
      evs = std::vector<real_t<T>>();

      std::vector<real_t<T>> eigvals_all(alpha.size());
      tridiagonal::tridiagonal_eigenvalues(alpha, beta, eigvals_all);
      if (this->find_maximum) {
        for (size_t i = 0; i < num_eigs_to_calculate; ++i) {
          evs.push_back(eigvals_all[eigvals_all.size() - i - 1]);
        }
      } else {
        for (size_t i = 0; i < num_eigs_to_calculate; ++i) {
          evs.push_back(eigvals_all[i]);
        }
      }

      const real_t<T> zero_threshold = std::numeric_limits<real_t<T>>::epsilon() * 1e1;
      if (beta.back() < zero_threshold) {
        itern = k;
        break;
      }

      util::normalize(u[k]);

      /*
       * Only break loop if convergence condition is met for all requied roots
       */
      bool break_cond = true;
      if (pevs.size() != evs.size()) {
        break_cond = false;
      } else {
        for (size_t iroot = 0; iroot < nroot; ++iroot) {
          const auto& ev = evs[iroot];
          const auto& pev = pevs[iroot];
          if (std::abs(ev - pev) >= std::min(std::abs(ev), std::abs(pev)) * this->eps) {
            break_cond = false;
            break;
          }
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
    eigvecs.resize(eigvalues.size());
    beta.back() = 0.0;

    eigvecs = compute_eigenvectors(alpha, beta, u, find_maximum, eigvalues.size());
    for (size_t i = 0; i < eigvalues.size(); ++i) {
      eigvalues[i] -= this->eigenvalue_offset;
    }

    return itern;
  }

  /**
   * @brief Executes Lanczos algorithm and stores the result into reference variables passed as arguments.
   *
   * @param [out] eigenvalues Eigenvalues. `eigenvalues[k]` stores the k-th eigenvalue.
   * @param [out] eigenvectors Eigenvectors. `eigenvectors[k][:]` stores the k-th eigenvector.
   */
  void run(std::vector<real_t<T>>& eigenvalues, std::vector<std::vector<T>>& eigenvectors) {
    this->iter_counts = std::vector<size_t>();
    eigenpair_manager::EigenPairManager<T> ep_manager(find_maximum, num_eigs);

    while (true) {
      std::vector<real_t<T>> eigenvalues_current;
      std::vector<std::vector<T>> eigenvectors_current;

      size_t nroot = std::min(num_eigs_per_iteration, this->matrix_size - ep_manager.size());

      size_t iter_count =
          this->run_iteration(eigenvalues_current, eigenvectors_current, nroot, ep_manager.getEigenvectors());
      this->iter_counts.push_back(iter_count);

      bool nothing_added = ep_manager.insertEigenpairs(eigenvalues_current, eigenvectors_current);

      if (nothing_added) {
        break;
      }
    }

    const auto& eigenpairs = ep_manager.getEigenpairs();
    eigenvalues = std::vector<real_t<T>>();
    eigenvalues.reserve(eigenpairs.size());
    eigenvectors = std::vector<std::vector<T>>();
    eigenvectors.reserve(eigenvectors.size());

    for (auto& eigenpair : eigenpairs) {
      eigenvalues.push_back(std::move(eigenpair.first));
      eigenvectors.push_back(std::move(eigenpair.second));
    }
  }

  /**
   * @brief Executes Lanczos algorithm and return result as a tuple.
   *
   * This function provides C++17 multiple-value-return interface.
   *
   * @return Eigenvalues. `eigenvalues[k]` stores the k-th eigenvalue.
   * @return Eigenvectors. `eigenvectors[k][:]` stores the k-th eigenvector.
   */
  std::tuple<std::vector<real_t<T>>, std::vector<std::vector<T>>> run() {
    std::vector<real_t<T>> eigenvalues;
    std::vector<std::vector<T>> eigenvectors;
    for (size_t i = 0; i < this->num_eigs; ++i) {
      eigenvectors.emplace_back(this->matrix_size);
    }

    this->run(eigenvalues, eigenvectors);

    return {eigenvalues, eigenvectors};
  }

  /**
   * @brief Executes Lanczos algorithm that calculate one eigenpair regardless of `num_eigs`.
   *
   * @param [out] eigenvalue Eigenvalue.
   * @param [out] eigenvector Eigenvector.
   */
  void run(real_t<T>& eigenvalue, std::vector<T>& eigenvector) {
    const size_t num_eigs_tmp = this->num_eigs;
    this->num_eigs = 1;

    std::vector<real_t<T>> eigenvalues(1);
    std::vector<std::vector<T>> eigenvectors(1);

    this->run(eigenvalues, eigenvectors);

    this->num_eigs = num_eigs_tmp;

    eigenvalue = eigenvalues[0];
    eigenvector = std::move(eigenvectors[0]);
  }

  /**
   * @brief Returns the latest iteration counts.
   */
  const std::vector<size_t>& getIterationCounts() const {
    return iter_counts;
  }
};

} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_H_ */
