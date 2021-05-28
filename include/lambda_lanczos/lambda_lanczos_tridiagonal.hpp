#ifndef LAMBDA_LANCZOS_TRIDIAGONAL_H_
#define LAMBDA_LANCZOS_TRIDIAGONAL_H_

#include <vector>
#include <complex>
#include "lambda_lanczos_util.hpp"


namespace lambda_lanczos { namespace tridiagonal {
/**
 * @brief Finds the number of eigenvalues of given tridiagonal matrix smaller than `c`.
 *
 * Algorithm from
 * Peter Arbenz et al. / "High Performance Algorithms for Structured Matrix Problems" /
 * Nova Science Publishers, Inc.
 */
template <typename T>
inline size_t num_of_eigs_smaller_than(T c,
                                       const std::vector<T>& alpha,
                                       const std::vector<T>& beta) {
  T q_i = alpha[0] - c;
  size_t count = 0;
  size_t m = alpha.size();

  if(q_i < 0){
    ++count;
  }

  for(size_t i = 1; i < m; ++i){
    q_i = alpha[i] - c - beta[i-1]*beta[i-1]/q_i;
    if(q_i < 0){
      ++count;
    }
    if(q_i == 0){
      q_i = util::minimum_effective_decimal<T>();
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
inline T tridiagonal_eigen_limit(const std::vector<T>& alpha,
                                 const std::vector<T>& beta) {
  T r = util::l1_norm(alpha);
  r += 2*util::l1_norm(beta);

  return r;
}


/**
 * @brief Finds the `m`th smaller eigenvalue of given tridiagonal matrix.
 */
template <typename T>
inline T find_mth_eigenvalue(const std::vector<T>& alpha,
                             const std::vector<T>& beta,
                             const size_t m,
                             const T eps) {
  T mid;
  T pmid = std::numeric_limits<T>::max();
  T r = tridiagonal_eigen_limit(alpha, beta);
  T lower = -r;
  T upper = r;

  while(upper-lower > std::min(std::abs(lower), std::abs(upper))*eps) {
    mid = (lower+upper)/2.0;

    if(num_of_eigs_smaller_than(mid, alpha, beta) >= m+1) {
      upper = mid;
    } else {
      lower = mid;
    }

    if(mid == pmid) {
      /* This avoids an infinite loop due to zero matrix */
      break;
    }
    pmid = mid;
  }

  return lower; // The "lower" almost equals the "upper" here.
}


/**
 * @brief Computes an eigenvector corresponding to given eigenvalue for given tri-diagonal matrix.
 */
template <typename T>
inline std::vector<T> tridiagonal_eigenvector(const std::vector<T>& alpha,
                                              const std::vector<T>& beta,
                                              T ev) {
  const auto m = alpha.size();
  std::vector<T> cv(m+1);
  cv[m] = 0.0;
  cv[m-1] = 1.0;

  for (size_t k = m-1; k-- > 0;) {
    cv[k] = ((ev - alpha[k + 1]) * cv[k + 1] - beta[k + 1] * cv[k + 2]) / beta[k];
  }

  util::normalize(cv);

  return cv;
}

}} // namespace lambda_lanczos::tridiagonal

#endif  /* LAMBDA_LANCZOS_TRIDIAGONAL_H_ */
