![CI](https://github.com/mrcdr/lambda-lanczos/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/mrcdr/lambda-lanczos/branch/master/graph/badge.svg)](https://codecov.io/gh/mrcdr/lambda-lanczos)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/mrcdr/lambda-lanczos)]()

Lambda Lanczos
===========

C++ adaptive and header-only Lanczos algorithm library

## Overview

**Lambda Lanczos** calculates the smallest or largest eigenvalue and
the corresponding eigenvector of a symmetric (Hermitian) matrix.

The characteristic feature is the matrix-vector multiplication routine used in
the Lanczos algorithm is adaptable:

```c++
#include <lambda_lanczos/lambda_lanczos.hpp>
using lambda_lanczos::LambdaLanczos;
/* Some include and using declarations are omitted */

void sample() {
  const int n = 3;
  double matrix[n][n] = { {2.0, 1.0, 1.0},
                          {1.0, 2.0, 1.0},
                          {1.0, 1.0, 2.0} };
  // Its eigenvalues are {4, 1, 1}

  /* Prepare matrix-vector multiplication routine used in Lanczos algorithm */
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
        out[i] += matrix[i][j]*in[j];
      }
    }
  };


  /* Execute Lanczos algorithm */
  LambdaLanczos<double> engine(mv_mul, n, true, 1); // Find 1 maximum eigenvalue
  vector<double> eigenvalues;
  vector<vector<double>> eigenvectors;
  engine.run(eigenvalues, eigenvectors);
  //// If C++17 is available, the following notation does the same thing:
  // auto [eigenvalues, eigenvectors] = engine.run()


  /* Print result */
  cout << "Eigenvalue: " << setprecision(16) << eigenvalues[0] << endl;
  cout << "Eigenvector:";
  for(int i = 0; i < n; i++) {
    cout << eigenvectors[0][i] << " ";
  }
  cout << endl;
}
```

This feature allows you to
- easily combine **Lambda Lanczos** with existing matrix libraries
(e.g. [Eigen](http://eigen.tuxfamily.org/index.php);
see a [sample code](https://github.com/mrcdr/lambda-lanczos/blob/master/src/samples/sample4_use_Eigen_library.cpp)).
- use a matrix whose elements are partially given,
  e.g. a sparse matrix whose non-zero elements are stored
  as a list of {row-index, column-index, value} tuples.

## Interfaces
Lambda Lanczos provides the following two interfaces:
### 1. Eigenvalue problem
`LambdaLanczos` class computes maximum (minimum) eigenvalues and
corresponding eigenvectors. Degenerate eigenvalues are took into account.
This class aims problems that require one or a few eigenpairs
(e.g. low-energy excitation of a quantum system).

### 2. Exponentiation
`Exponentiator` class computes the following type of matrix-vector multiplication:

$$\boldsymbol{v}'=e^{\delta A} \boldsymbol{v},$$

where $A$ is a symmetric (Hermitian) matrix and $\delta$ is a scalar parameter.
This class is based on the same theory of the Lanczos algorithm (Krylov subspace method).

As an application, this class may be used for
time evolution of a quantum many-body system:

$$ \ket{\psi(t+\Delta t)} = e^{-iH\Delta t} \ket{\psi(t)},$$

and more sophisticated algorithms, such as [TDVP](https://arxiv.org/abs/1408.5056) and other tensor networks.



## Sample programs
See [here](https://github.com/mrcdr/lambda-lanczos/tree/master/src/samples).

## API reference
[API reference](https://mrcdr.github.io/lib-docs/lambda-lanczos/)

## Requirement

C++11 compatible environment

## Dependencies
**Lambda Lanczos** itself does not depend on any libraries.

In order to run tests, [Google Test](https://github.com/google/googletest) is required.

## Installation

**Lambda Lanczos** is a header-only library.
So the installation step is as follows:

1. Clone or download the latest version from [Github](https://github.com/mrcdr/lambda-lanczos/).
2. Place the `include/lambda_lanczos` directory anywhere your project can find.

## License

[MIT](https://github.com/mrcdr/lambda-lanczos/blob/master/LICENSE)

## Author

[mrcdr](https://github.com/mrcdr)
