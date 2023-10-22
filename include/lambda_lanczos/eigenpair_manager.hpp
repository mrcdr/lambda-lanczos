#ifndef LAMBDA_LANCZOS_EIGENPAIR_MANAGER_H_
#define LAMBDA_LANCZOS_EIGENPAIR_MANAGER_H_

#include <functional>
#include <vector>

#include "lambda_lanczos_util.hpp"

namespace lambda_lanczos {
namespace eigenpair_manager {

/**
 * @brief Class to manage calculated eigenpairs.
 *
 * This class does:
 * (1) keep only N largest (smallest) eigenvalues and correspoinding eigenvectors.
 * (2) add new eigenpairs and examine whether some of them are kept or all of them are dropped by (1)
 *     (which determine whether more Lanczos iterations are required).
 * (3) give reference to eigenvectors.
 */
template <typename T>
class EigenPairManager {
 private:
  template <typename n_type>
  using real_t = util::real_t<n_type>;

  const bool find_maximum;
  const size_t num_eigs;

  std::multimap<real_t<T>, std::vector<T>, std::function<bool(real_t<T>, real_t<T>)>> eigenpairs;
  // This super long definition is required because the default comparator type is std::less<?>.

 public:
  EigenPairManager(bool find_maximum, size_t num_eigs) : find_maximum(find_maximum), num_eigs(num_eigs) {
    std::function<bool(real_t<T>, real_t<T>)> comp;
    if (find_maximum) {
      comp = std::greater<real_t<T>>();
    } else {
      comp = std::less<real_t<T>>();
    }

    std::vector<std::pair<real_t<T>, std::vector<T>>> dummy;
    this->eigenpairs = std::multimap<real_t<T>, std::vector<T>, std::function<bool(real_t<T>, real_t<T>)>>(
        dummy.begin(), dummy.end(), comp);
    // This super long definition is required because the default comparator type is std::less<?>.
  }

  size_t size() const {
    return eigenpairs.size();
  }

  bool insertEigenpairs(std::vector<real_t<T>>& eigenvalues, std::vector<std::vector<T>>& eigenvectors) {
    assert(eigenvalues.size() == eigenvectors.size());

    bool nothing_added = true;
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
      auto inserted = eigenpairs.emplace(std::move(eigenvalues[i]), std::move(eigenvectors[i]));
      auto last = eigenpairs.end();
      last--;
      if (eigenpairs.size() > num_eigs) {
        if (inserted != last) {  // If the eigenpair is not inserted to the tail
          nothing_added = false;
        }
        eigenpairs.erase(last);
      } else {
        nothing_added = false;
      }
    }

    return nothing_added;
  }

  lambda_lanczos::util::MapValueIterable<decltype(eigenpairs)> getEigenvectors() const {
    return lambda_lanczos::util::MapValueIterable<decltype(eigenpairs)>(eigenpairs);
  }

  decltype(eigenpairs)& getEigenpairs() {
    return eigenpairs;
  };
};

} /* namespace eigenpair_manager */
} /* namespace lambda_lanczos */
#endif /* LAMBDA_LANCZOS_EIGENPAIR_MANAGER_H_ */