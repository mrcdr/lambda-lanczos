#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <lambda_lanczos/lambda_lanczos.hpp>

using std::cout;
using std::endl;
using std::setprecision;
using lambda_lanczos::LambdaLanczos;

template<typename T>
using vector = std::vector<T>;

void test1() {
  const int n = 3;
  double matrix[n][n] = { {-2.0, -1.0, -1.0},
                          {-1.0, -2.0, -1.0},
			  {-1.0, -1.0, -2.0} };
  /* Its eigenvalues are {-4, -1, -1} */  

  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      for(int j = 0;j < n;j++) {
	out[i] += matrix[i][j]*in[j];
      }
    } 
  };

  LambdaLanczos engine(matmul, n);
  double eigvalue;
  vector<double> eigvec(n);
  int itern  = engine.run(eigvalue, eigvec);

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigvalue << endl;
  cout << "Eigen vector:";
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << " ";
  }
  cout << endl;
}

void test2() {
  const int n = 100;

  double offset = -10.0;
  auto matmul = [&](const vector<double>& in, vector<double>& out) {
    for(int i = 0;i < n;i++) {
      out[i] += offset*in[i];
    }
    
    for(int i = 0;i < n-1;i++) {
      out[i] += -1.0*in[i+1];
      out[i+1] += -1.0*in[i];
    }

//    out[0] += -1.0*in[n-1]; // This corresponds to 
//    out[n-1] += -1.0*in[0]; // periodic boundary condition
  };
  /*
    This lambda is equivalent to applying
    following n by n matrix ("off" means "offset"):
    
    off  -1   0       ..      0
     -1 off  -1       ..      0
      0  -1 off       ..      0
      0     ..        ..      0
      0     ..      off  -1   0
      0     ..       -1 off  -1
      0     ..        0  -1 off

      Its smallest eigenvalue is -2*cos(pi/(n+1)) + offset.
      The "offset" is required to make the absolute value of
      the smallest eigenvalue maximum.
   */

  LambdaLanczos engine(matmul, n);
  engine.eps = 1e-14;
  double eigvalue;
  vector<double> eigvec(n);
  int itern  = engine.run(eigvalue, eigvec);  
  eigvalue -= offset;

  cout << "Iteration count: " << itern << endl;
  cout << "Eigen value: " << setprecision(16) << eigvalue << endl;  
  cout << "Eigen vector:" << endl;
  for(int i = 0;i < n;i++) {
    cout << eigvec[i] << endl;
  }
  cout << endl;
}

int main() {
  test1();
  test2();

  return EXIT_SUCCESS;
}
