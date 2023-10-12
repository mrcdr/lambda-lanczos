#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_IMPL_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_IMPL_H_

#include <algorithm>
#include <complex>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>

#include "lambda_lanczos_util.hpp"

namespace lambda_lanczos {
namespace tridiagonal_impl {
/**
 * @brief Finds the number of eigenvalues of given tridiagonal matrix smaller than `c`.
 *
 * Algorithm from
 * Peter Arbenz et al. / "High Performance Algorithms for Structured Matrix Problems" /
 * Nova Science Publishers, Inc.
 */
template <typename T>
inline size_t num_of_eigs_smaller_than(T c, const std::vector<T>& alpha, const std::vector<T>& beta) {
  T q_i = alpha[0] - c;
  size_t count = 0;
  size_t m = alpha.size();

  if (q_i < 0) {
    ++count;
  }

  for (size_t i = 1; i < m; ++i) {
    q_i = alpha[i] - c - beta[i - 1] * beta[i - 1] / q_i;
    if (q_i < 0) {
      ++count;
    }
    if (q_i == 0) {
      q_i = std::numeric_limits<T>::epsilon();
    }
  }

  return count;
}

/**
 * @brief Computes the upper bound of the absolute value of eigenvalues by Gerschgorin theorem.
 *
 * This routine gives a rough upper bound,
 * but it is sufficient because the bisection routine using
 * the upper bound converges exponentially.
 */
template <typename T>
inline T tridiagonal_eigen_limit(const std::vector<T>& alpha, const std::vector<T>& beta) {
  T r = util::m_norm(alpha);
  r += 2 * util::m_norm(beta);

  return r;
}

/**
 * @brief Finds the `m`th smaller eigenvalue of given tridiagonal matrix.
 */
template <typename T>
inline T find_mth_eigenvalue(const std::vector<T>& alpha, const std::vector<T>& beta, const size_t m) {
  T mid;
  T pmid = std::numeric_limits<T>::max();
  T r = tridiagonal_eigen_limit(alpha, beta);
  T lower = -r;
  T upper = r;

  while (upper - lower > std::min(std::abs(lower), std::abs(upper)) * std::numeric_limits<T>::epsilon()) {
    mid = (lower + upper) * T(0.5);

    if (num_of_eigs_smaller_than(mid, alpha, beta) >= m + 1) {
      upper = mid;
    } else {
      lower = mid;
    }

    if (mid == pmid) {
      /* This avoids an infinite loop due to zero matrix */
      break;
    }
    pmid = mid;
  }

  return lower;  // The "lower" almost equals the "upper" here.
}

/**
 * @brief Computes an eigenvector corresponding to given eigenvalue for given tri-diagonal matrix.
 */
template <typename T>
inline std::vector<T> compute_tridiagonal_eigenvector_from_eigenvalue(const std::vector<T>& alpha,
                                                                      const std::vector<T>& beta,
                                                                      const size_t index,
                                                                      const T ev) {
  (void)index;  // Unused declaration for compiler

  const auto m = alpha.size();
  std::vector<T> cv(m);

  cv[m - 1] = 1.0;

  if (m >= 2) {
    cv[m - 2] = (ev - alpha[m - 1]) * cv[m - 1] / beta[m - 2];
    for (size_t k = m - 2; k-- > 0;) {
      cv[k] = ((ev - alpha[k + 1]) * cv[k + 1] - beta[k + 1] * cv[k + 2]) / beta[k];
    }
  }

  util::normalize(cv);

  return cv;
}

/**
 * @brief Computes all eigenpairs (eigenvalues and eigenvectors) for given tri-diagonal matrix.
 */
template <typename T>
inline void tridiagonal_eigenpairs_bisection(const std::vector<T>& alpha,
                                             const std::vector<T>& beta,
                                             std::vector<T>& eigenvalues,
                                             std::vector<std::vector<T>>& eigenvectors) {
  const size_t n = alpha.size();
  eigenvalues.resize(n);
  eigenvectors.resize(n);

  for (size_t j = 0; j < n; ++j) {
    eigenvalues[j] = lambda_lanczos::tridiagonal_impl::find_mth_eigenvalue(alpha, beta, j);
    eigenvectors[j] = lambda_lanczos::tridiagonal_impl::compute_tridiagonal_eigenvector_from_eigenvalue(
        alpha, beta, j, eigenvalues[j]);
  }
}

/**
 * @brief Calculates the cosine and sine of Givens rotation that eliminates a specific element (see detail).
 *
 * @details
 * This function calculates the cosine and sine that eliminate the element b, i.e. that satisfy
 * @verbatim
   ( c  s)(a) = (x)
   (-s  c)(b) = (0),
 * @endverbatim
 * where x >= 0.
 *
 * @param [in] a Input element to remain non-zero.
 * @param [in] b Input element to be eliminated.
 * @return Pair (c, s).
 */
template <typename T>
inline std::pair<T, T> calc_givens_cs(T a, T b) {
  if (b == 0) {
    return std::make_pair(1.0, 0.0);
  }

  if (a == 0) {
    return std::make_pair(0.0, 1.0);
  }

  T x = std::sqrt(a * a + b * b);
  T c = a / x;
  T s = b / x;

  return std::make_pair(c, s);
}

/**
 * @brief Performs an implicit shift QR step A = Z^T A Z on given sub tridiagonal matrix A.
 * @param [in, out] alpha Diagonal elements of the full tridiagonal matrix.
 * @param [in, out] beta Subdiagonal elements of the full tridiagonal matrix.
 * @param [in, out] q A "matrix" Q that will be overwritten as Q = QZ if `rotate_matrix` is true. See note for details.
 * @param offset The first index of the submatrix.
 * @param nsub The size of the submatrix.
 * @param rotate_matrix True to apply Given's rotations to the matrix `q`. If false, the matrix `q` won't be accessed.
 *
 * @note A series of the QR steps produces an eigenvector "matrix" q
 * that stores the k-th eigenvector as `q[k][:]`.
 * This definition differs from a usual mathematical sense (i.e., Q_{:,k} specifies the k-th eigenvector).
 */
template <typename T>
inline void isqr_step(std::vector<T>& alpha,
                      std::vector<T>& beta,
                      std::vector<std::vector<T>>& q,
                      size_t offset,
                      size_t nsub,
                      bool rotate_matrix) {
  using lambda_lanczos::util::sgn;
  using std::pow;
  using std::sqrt;

  if (nsub == 1) {
    return;
  }

  T d = (alpha[offset + nsub - 2] - alpha[offset + nsub - 1]) / (2 * beta[offset + nsub - 2]);
  T mu = alpha[offset + nsub - 1] - beta[offset + nsub - 2] / (d + sgn(d) * sqrt(d * d + T(1)));
  T x = alpha[offset + 0] - mu;

  T s = 1.0;
  T c = 1.0;
  T p = 0.0;

  for (size_t k = 0; k < nsub - 1; ++k) {
    T z = s * beta[offset + k];
    T beta_prev = c * beta[offset + k];

    auto cs = calc_givens_cs(x, z);
    c = std::get<0>(cs);
    s = std::get<1>(cs);

    if (k > 0) {
      beta[offset + k - 1] = sqrt(x * x + z * z);
    }
    T u = ((alpha[offset + k + 1] - alpha[offset + k] + p) * s + T(2) * c * beta_prev);
    alpha[offset + k] = alpha[offset + k] - p + s * u;
    p = s * u;
    x = c * u - beta_prev;

    // Keep in mind that q[k][j] is the jth element of the kth eigenvector.
    // This means an eigenvector is stored as a ROW of the matrix q
    // in the sense of mathematical notation.
    if (rotate_matrix) {
      for (size_t j = 0; j < alpha.size(); ++j) {
        auto v0 = q[offset + k][j];
        auto v1 = q[offset + k + 1][j];

        q[offset + k][j] = c * v0 + s * v1;
        q[offset + k + 1][j] = -s * v0 + c * v1;
      }
    }
  }

  alpha[offset + nsub - 1] = alpha[offset + nsub - 1] - p;
  beta[offset + nsub - 2] = x;
}

/**
 * @brief Find sub-tridiagonal matrix that remains non-diagonal.
 *
 * @details
 * This function does the following two things:
 * 1. Overwrite small elements with zero,
 * 2. Find subspace that remains non-diagonal.
 * The dimension of the resulting sub-matrix is submatrix_last - submatrix_first + 1.
 *
 * @param [in] alpha Diagonal elements of the full tridiagonal matrix.
 * @param [in,out] beta Sub-diagonal elements of the full tridiagonal matrix.
 * @param [out] submatrix_first Index to point the first element of the sub-tridiagonal matrix.
 * @param [in,out] submatrix_last Index to point the last element of the sub-tridiagonal matrix.
 */
template <typename T>
inline void find_subspace(const std::vector<T>& alpha,
                          std::vector<T>& beta,
                          size_t& submatrix_first,
                          size_t& submatrix_last) {
  const T eps = std::numeric_limits<T>::epsilon() * 0.5;
  const T safe_min = std::numeric_limits<T>::min();
  const size_t n = alpha.size();

  /* Overwrite small elements with zero  */
  for (size_t i = 0; i < n - 1; ++i) {
    if (std::abs(beta[i]) < std::sqrt(std::abs(alpha[i]) * std::abs(alpha[i + 1])) * eps + safe_min) {
      beta[i] = 0;
    }
  }

  /* Find subspace */
  while (submatrix_last > 0 && beta[submatrix_last - 1] == 0) {
    submatrix_last--;
  }
  submatrix_first = submatrix_last;
  while (submatrix_first > 0 && beta[submatrix_first - 1] != 0) {
    submatrix_first--;
  }
}

/**
 * @brief Computes all eigenpairs (eigenvalues and eigenvectors) for given tri-diagonal matrix
 * using the Implicitly Shifted QR algorithm.
 *
 * @param [in] alpha Diagonal elements of the full tridiagonal matrix.
 * @param [in] beta Sub-diagonal elements of the full tridiagonal matrix.
 * @param [out] eigenvalues Eigenvalues.
 * @param [out] eigenvectors Eigenvectors. The k-th eigenvector will be stored in `eigenvectors[k]`.
 * @param [in] compute_eigenvector True to calculate eigenvectors. If false, `eigenvectors` won't be accessed.
 *
 * @return Count of forced breaks due to unconvergence.
 */
template <typename T>
inline size_t tridiagonal_eigenpairs(const std::vector<T>& alpha,
                                     const std::vector<T>& beta,
                                     std::vector<T>& eigenvalues,
                                     std::vector<std::vector<T>>& eigenvectors,
                                     bool compute_eigenvector = true) {
  const size_t n = alpha.size();

  auto alpha_work = alpha;
  auto beta_work = beta;

  /* Prepare an identity matrix to be transformed into an eigenvector matrix */
  if (compute_eigenvector) {
    lambda_lanczos::util::initAsIdentity(eigenvectors, n);
  }

  size_t unconverged_count = 0;
  size_t submatrix_last_prev = n - 1;

  size_t loop_count = 1;
  while (true) {
    size_t submatrix_last = submatrix_last_prev;
    size_t submatrix_first;
    find_subspace(alpha_work, beta_work, submatrix_first, submatrix_last);

    const size_t nsub = submatrix_last - submatrix_first + 1;
    const size_t max_loop_count = nsub * 50;

    if (submatrix_last > 0) {
      isqr_step(alpha_work, beta_work, eigenvectors, submatrix_first, nsub, compute_eigenvector);
    } else {
      break;
    }

    if (submatrix_last == submatrix_last_prev) {
      if (loop_count > max_loop_count) {
        submatrix_last_prev = submatrix_first;
        unconverged_count++;
        loop_count = 1;
      } else {
        loop_count++;
      }
    } else {
      loop_count = 1;
      submatrix_last_prev = submatrix_last;
    }
  }

  eigenvalues = alpha_work;

  lambda_lanczos::util::sort_eigenpairs(eigenvalues, eigenvectors, compute_eigenvector);

  return unconverged_count;
}

/**
 * @brief Computes all eigenvalues for given tri-diagonal matrix
 * using the Implicitly Shifted QR algorithm.
 *
 * @param [in] alpha Diagonal elements of the full tridiagonal matrix.
 * @param [in] beta Sub-diagonal elements of the full tridiagonal matrix.
 * @param [out] eigenvalues Eigenvalues.
 *
 * @return Count of forced breaks due to unconvergence.
 */
template <typename T>
inline size_t tridiagonal_eigenvalues(const std::vector<T>& alpha,
                                      const std::vector<T>& beta,
                                      std::vector<T>& eigenvalues) {
  std::vector<std::vector<T>> dummy_eigenvectors;
  return tridiagonal_eigenpairs(alpha, beta, eigenvalues, dummy_eigenvectors, false);
}

} /* namespace tridiagonal_impl */
} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_TRIDIAGONAL_IMPL_H_ */
