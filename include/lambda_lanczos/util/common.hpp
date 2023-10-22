#ifndef LAMBDA_LANCZOS_UTIL_COMMON_H_
#define LAMBDA_LANCZOS_UTIL_COMMON_H_

#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

namespace lambda_lanczos {
namespace util {

/**
 * @brief Iterator for a container of tuples to iterate over the I-th tuple elements.
 */
template <size_t I, typename container_type>
class TupleViewIterator {
 private:
  typename container_type::const_iterator iter;

 public:
  TupleViewIterator(const typename container_type::const_iterator& iter) : iter(iter) {}

  const typename std::tuple_element<I, typename container_type::value_type>::type& operator*() const {
    return std::get<I>(*iter);
  }

  bool operator==(const TupleViewIterator& obj) const {
    return this->iter == obj.iter;
  }

  bool operator!=(const TupleViewIterator& obj) const {
    return this->iter != obj.iter;
  }

  TupleViewIterator& operator++() {
    this->iter++;
    return *this;
  }

  TupleViewIterator operator++(int dummy) {
    auto before = *this;
    this->iter++;
    return before;
  }
};

/**
 * @brief Iterator for a map to iterate over its values.
 */
template <typename map_type>
using MapValueIterator = TupleViewIterator<1, map_type>;

template <typename map_type>
class MapValueIterable {
 private:
  const typename map_type::const_iterator itr_cbegin;
  const typename map_type::const_iterator itr_cend;

 public:
  MapValueIterable(const map_type& map) : itr_cbegin(map.cbegin()), itr_cend(map.cend()) {}

  MapValueIterator<map_type> cbegin() const {
    return itr_cbegin;
  }

  MapValueIterator<map_type> cend() const {
    return itr_cend;
  }
};

/**
 * @brief Template class to map specific types. See #real_t<T> for usage.
 *
 */
template <typename T>
struct realTypeMap {
  typedef T type;
};

template <typename T>
struct realTypeMap<std::complex<T>> {
  typedef T type;
};

/**
 * @brief Type mapper from `T` to real type of `T`.
 *
 * By default, `real_t<T>` returns `T`.
 * However `real_t<complex<T>>` returns `T`.
 * Usage example: This function returns a real number even if `T` is complex:
 * @code
 * template <typename T>
 * inline real_t<T> norm(const std::vector<T>& vec);
 * @endcode
 */
template <typename T>
using real_t = typename realTypeMap<T>::type;

/**
 * @brief Complex conjugate template.
 *
 * This structure is required because partial specialization of
 * function template is not allowed in C++.
 *
 * Use #typed_conj in practice.
 */
template <typename T>
struct TypedConjugate {
  static T invoke(const T& val) {
    return val;
  }
};

template <typename T>
struct TypedConjugate<std::complex<T>> {
  static std::complex<T> invoke(const std::complex<T> val) {
    return std::conj(val);
  }
};

/**
 * @brief Complex conjugate with type.
 * This function returns the argument itself for real type,
 * and returns its complex conjugate for complex type.
 */
template <typename T>
inline T typed_conj(const T& val) {
  return TypedConjugate<T>::invoke(val);
}

/**
 * @brief Sorts eigenvalues and eigenvectors with respect to given predicate.
 *
 * @note This function changes the memory location of the eigenpairs.
 */
template <typename T>
inline void sort_eigenpairs(std::vector<real_t<T>>& eigenvalues,
                            std::vector<std::vector<T>>& eigenvectors,
                            bool sort_eigenvector,
                            const std::function<bool(real_t<T>, real_t<T>)> predicate = std::less<real_t<T>>()) {
  using ev_index_t = std::pair<real_t<T>, size_t>;
  std::vector<ev_index_t> ev_index_pairs;
  ev_index_pairs.reserve(eigenvalues.size());
  for (size_t i = 0; i < eigenvalues.size(); ++i) {
    ev_index_pairs.emplace_back(eigenvalues[i], i);
  }

  std::sort(ev_index_pairs.begin(), ev_index_pairs.end(), [&predicate](const ev_index_t& x, const ev_index_t& y) {
    return predicate(x.first, y.first);
  });

  std::vector<real_t<T>> eigenvalues_new;
  eigenvalues_new.reserve(eigenvalues.size());
  for (const auto& ev_index : ev_index_pairs) {
    size_t k = ev_index.second;
    eigenvalues_new.emplace_back(eigenvalues[k]);
  }
  eigenvalues = std::move(eigenvalues_new);

  if (sort_eigenvector) {
    std::vector<std::vector<T>> eigenvectors_new;
    eigenvectors_new.reserve(eigenvalues.size());
    for (const auto& ev_index : ev_index_pairs) {
      size_t k = ev_index.second;
      eigenvectors_new.push_back(std::move(eigenvectors[k]));
    }
    eigenvectors = std::move(eigenvectors_new);
  }
}

/**
 * @brief Returns the significant decimal digits of type T.
 *
 */
template <typename T>
inline constexpr int sig_decimal_digit() {
  return (int)(std::numeric_limits<T>::digits * log10(std::numeric_limits<T>::radix));
}

template <typename T>
inline constexpr T minimum_effective_decimal() {
  return pow(10, -sig_decimal_digit<T>());
}

/**
 * @brief Return the sign of given value.
 * @details If 0 is given, this function returns +1.
 */
template <typename T>
T sgn(T val) {
  if (val >= 0) {
    return (T)1;
  } else {
    return (T)(-1);
  }
}

/**
 * @brief Returns string representation of given vector.
 */
template <typename T>
std::string vectorToString(const std::vector<T>& vec, std::string delimiter = " ") {
  std::stringstream ss;

  for (const auto& elem : vec) {
    ss << elem << delimiter;
  }

  /* Remove the last space */
  std::string result = ss.str();
  if (!result.empty()) {
    result.pop_back();
  }

  return result;
}

} /* namespace util */
} /* namespace lambda_lanczos */

#endif /* LAMBDA_LANCZOS_UTIL_COMMON_H_ */
