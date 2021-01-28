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
  // its eigenvalues are {4, 1, 1}

  /* Prepare matrix-vector multiplication routine used in Lanczos algorithm */
  auto mv_mul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
        out[i] += matrix[i][j]*in[j];
      }
    }
  };


  /* Execute Lanczos algorithm */
  LambdaLanczos<double> engine(mv_mul, n, true); // true means to calculate the largest eigenvalue.
  double eigenvalue;
  vector<double> eigenvector(n);
  int itern = engine.run(eigenvalue, eigenvector);
  //// If C++17 is available, the following notation does the same thing:
  // auto [eigenvalue, eigenvector, itern] = engine.run()


  /* Print result */
  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigenvalue << endl;
  cout << "Eigen vector:";
  for(int i = 0;i < n;i++) {
    cout << eigenvector[i] << " ";
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

For detailed specs, see [API reference](https://mrcdr.github.io/lib-docs/lambda-lanczos/).

## Sample programs
See [here](https://github.com/mrcdr/lambda-lanczos/tree/master/src/samples).

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

## Use Lambda Lanczos correctly
### What is `eigenvalue_offset`?
  The Lanczos algorithm can find the largest magnitude eigenvalue, so **you must ensure
  the maximum/minimum eigenvalue to be calculated has the largest magnitude**.
  
  For any n by n matrix A, the upper bound *r* of the magnitudes of the eigenvalues can be
  determined by Gershgorin theorem:

  <img src="https://latex.codecogs.com/gif.latex?\large&space;r=\max_{i=1..n}\left\{\sum_{j=1}^n|A_{ij}|\right\}"/>
  
  or

  <img src="https://latex.codecogs.com/gif.latex?\large&space;r=\max_{j=1..n}\left\{\sum_{i=1}^n|A_{ij}|\right\}"/>

  So corresponding code would be like [this](https://github.com/mrcdr/lambda-lanczos/blob/master/src/determine_eigenvalue_offset/determine_eigenvalue_offset.cpp).
  If you want to calculate the maximum eigenvalue, you should use `eigenvalue_offset = r`. To calculate the minimum eigenvalue `eigenvalue_offset = -r`.

  More information about `eigenvalue_offset` is available [here](https://github.com/mrcdr/lambda-lanczos/wiki/More-about-eigenvalue_offset).

## License

[MIT](https://github.com/mrcdr/lambda-lanczos/blob/master/LICENSE)

## Author

[mrcdr](https://github.com/mrcdr)
