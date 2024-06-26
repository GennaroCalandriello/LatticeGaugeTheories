#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include "const.h"
#include "distributions.h"
#include "paulimatrices.h"
#include "su2.h"

using namespace std;
using Complex = complex<double>;

// // implement parametrized contstructor
// SU2Matrix::SU2Matrix(Matrix &elements) : elements_(elements) {}

// // implement destructor
// SU2Matrix::~SU2Matrix() {}

SU2Matrix SU2Matrix::operator+(const SU2Matrix &other) const {
  Matrix result(su2, vector<Complex>(su2));
  for (int i = 0; i < su2; i++) {
    for (int j = 0; j < su2; j++) {
      result[i][j] = elements_[i][j] + other.elements_[i][j];
    }
  }
  return SU2Matrix(result);
}

SU2Matrix SU2Matrix::operator-(const SU2Matrix &other) const {
  Matrix result(su2, vector<Complex>(su2));
  for (int i = 0; i < su2; i++) {
    for (int j = 0; j < su2; j++) {
      result[i][j] = elements_[i][j] - other.elements_[i][j];
    }
  }
  return SU2Matrix(result);
}

SU2Matrix SU2Matrix::operator*(const SU2Matrix &other) const {
  Matrix result(su2, vector<Complex>(su2));
  for (int i = 0; i < su2; i++) {
    for (int j = 0; j < su2; j++) {
      for (int k = 0; k < su2; k++) {
        result[i][j] += elements_[i][k] * other.elements_[k][j];
      }
    }
  }
  return SU2Matrix(result);
}

SU2Matrix SU2Matrix::operator*(const double &rhs) const {
  // sintax is Matrix*complex
  SU2Matrix result;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result(i, j) = this->elements_[i][j] * rhs;
    }
  }

  return result;
}
SU2Matrix SU2Matrix::operator*(const Complex &rhs) const {
  // sintax is Matrix*complex
  SU2Matrix result;

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      result(i, j) = this->elements_[i][j] * rhs;
    }
  }

  return result;
}
SU2Matrix &SU2Matrix::operator+=(const SU2Matrix &rhs) {
  for (int i = 0; i < su2; ++i) {
    for (int j = 0; j < su2; ++j) {
      elements_[i][j] += rhs.elements_[i][j];
    }
  }
  return *this;
}

SU2Matrix &SU2Matrix::operator*=(const SU2Matrix &rhs) {
  Matrix result = {};
  for (int i = 0; i < su2; ++i) {
    for (int j = 0; j < su2; ++j) {
      for (int k = 0; k < su2; ++k) {
        result[i][j] += elements_[i][k] * rhs.elements_[k][j];
      }
    }
  }
  elements_ = result;
  return *this;
}
SU2Matrix &SU2Matrix::operator/=(const Complex &rhs) {
  for (int i = 0; i < su2; ++i) {
    for (int j = 0; j < su2; ++j) {
      elements_[i][j] /= rhs;
    }
  }
  return *this;
}

SU2Matrix &SU2Matrix::operator/=(const double &rhs) {
  for (int i = 0; i < su2; ++i) {
    for (int j = 0; j < su2; ++j) {
      elements_[i][j] /= rhs;
    }
  }
  return *this;
}

Complex &SU2Matrix::operator()(int row, int col) {
  if (row >= 0 && row < su2 && col >= 0 && col < su2) {
    return elements_[row][col];

  } else {
    throw out_of_range("Invalid row or column index");
  }
}

void SU2Matrix::print() const {
  for (const auto &row : elements_) {
    for (const auto &element : row) {
      cout << element << " ";
    }
    cout << endl;
  }
}

SU2Matrix SU2Matrix::conjT() const {
  Matrix result(su2, vector<Complex>(su2));
  for (int i = 0; i < su2; i++) {
    for (int j = 0; j < su2; j++) {
      result[i][j] = std::conj(elements_[j][i]);
    }
  }
  return SU2Matrix(result);
}

Complex SU2Matrix::det() const {
  const auto &a = elements_[0][0];
  const auto &b = elements_[0][1];
  const auto &d = elements_[1][0];
  const auto &e = elements_[1][1];

  return (a * e - d * b);
}

Complex SU2Matrix::tr() const {
  Complex trace = 0;
  for (int i = 0; i < su2; i++) {
    trace += elements_[i][i];
  }
  return trace;
}

double SU2Matrix::reTr() const {
  double realTrace = 0;
  for (int i = 0; i < su2; i++) {
    realTrace += elements_[i][i].real();
  }
  return realTrace;
}

SU2Matrix IdentityMat() {
  SU2Matrix identity;
  for (int i = 0; i < su2; i++) {
    identity(i, i) = Complex(1, 0);
  }
  return identity;
}

SU2Matrix su2_generator() {

  //     srand (clock());

  SU2Matrix su2_matrix;

  double r0 = uniform_(-0.5, 0.499);
  int sign = 0;
  if (r0 >= 0) {
    sign = 1;
  } else {
    sign = -1;
  }

  double x0 = sign * sqrt(1 - pow(epsilon, 2));
  vector<double> r(3, 0);
  vector<double> r_norm(3, 0);
  SU2Matrix I = IdentityMat();

  for (int i = 0; i < 3; i++) {
    r[i] = uniform_(0, 0.999) - 0.5;
  }

  double norm = sqrt(pow(r[0], 2) + pow(r[1], 2) + pow(r[2], 2));

  for (int i = 0; i < 3; i++) {
    r_norm[i] = epsilon * r[i] / norm;
  }

  for (int a = 0; a < 2; a++) {
    for (int b = 0; b < 2; b++) {
      complex<double> temp = x0 * I(a, b) + 1i * r_norm[0] * s_x[a][b] +
                             1i * r_norm[1] * s_y[a][b] +
                             1i * r_norm[2] * s_z[a][b];

      double realpart = temp.real();
      double imagpart = temp.imag();

      su2_matrix(a, b) = Complex(realpart, imagpart);
    }
  }

  return su2_matrix;
}

// int main() {
//   SU2Matrix s1 = su2_generator();
//   SU2Matrix s2 = su2_generator();
//   SU2Matrix s3 = su2_generator();
//   SU2Matrix s4 = su2_generator();
//   SU2Matrix s5 = su2_generator();
//   SU2Matrix s6 = su2_generator();

//   SU2Matrix prod = s1 * s2 * s3 + s4 * s5 * s6;
//   prod.print();
//   complex<double> det = sqrt(prod.det());
//   prod /= det;
//   cout << "det" << prod.det() << endl;
//   prod.print();
//   return 0;
// }