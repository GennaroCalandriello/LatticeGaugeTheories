// #include "Eigen/Dense"
#include <array>
#include <complex>
#include <ctime>
#include <iostream>
#include <random>

#include "SU3Matrix.h"
#include "const.h"
#include "distributions.h"
#include "su2.h"

using namespace std;
using Complex = complex<double>;

// g++ -g3 -Wall SU3Matrix.cpp su2.cpp distributions.cpp -o SU3Matrix.exe
// g++ -g3 -Wall -I/eigen/Eigen SU3Matrix.cpp su2.cpp distributions.cpp -o
// SU3Matrix.exe

// implement parametrized constructor
// SU3Matrix::SU3Matrix(const Matrix &elements) : elements_(elements) {}

// implement destructor
// SU3Matrix::~SU3Matrix() {}

SU3Matrix SU3Matrix::operator+(const SU3Matrix &other) const {
  Matrix result(su3, vector<Complex>(su3));
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      result[i][j] = elements_[i][j] + other.elements_[i][j];
    }
  }
  return SU3Matrix(result);
}

SU3Matrix SU3Matrix::operator-(const SU3Matrix &other) const {
  Matrix result(su3, vector<Complex>(su3));
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++)
      result[i][j] = elements_[i][j] - other.elements_[i][j];
  }
  return SU3Matrix(result);
}

SU3Matrix SU3Matrix::operator*(const SU3Matrix &other) const {
  Matrix result(su3, vector<Complex>(su3));
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      for (int k = 0; k < su3; k++) {
        result[i][j] += elements_[i][k] * other.elements_[k][j];
      }
    }
  }
  return SU3Matrix(result);
}

SU3Matrix SU3Matrix::operator*(const Complex &rhs) const {
  // sintax is Matrix*complex
  SU3Matrix result;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result(i, j) = this->elements_[i][j] * rhs;
    }
  }

  return result;
}

SU3Matrix SU3Matrix::operator/(const Complex &other) const {
  Matrix result(su3, vector<Complex>(su3));
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      result[i][j] = elements_[i][j] / other;
    }
  }
  return SU3Matrix(result);
}

SU3Matrix SU3Matrix::operator+(const Complex &other) const {
  Matrix result(su3, vector<Complex>(su3));
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      result[i][j] = elements_[i][j] + other;
    }
  }
  return SU3Matrix(result);
}

SU3Matrix SU3Matrix::operator-(const Complex &other) const {
  Matrix result(su3, vector<Complex>(su3));
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      result[i][j] = elements_[i][j] - other;
    }
  }
  return SU3Matrix(result);
}

// Inverse of a matrix
SU3Matrix SU3Matrix::inv() {
  Complex a = elements_[0][0];
  Complex b = elements_[0][1];
  Complex c = elements_[0][2];
  Complex d = elements_[1][0];
  Complex e = elements_[1][1];
  Complex f = elements_[1][2];
  Complex g = elements_[2][0];
  Complex h = elements_[2][1];
  Complex i = elements_[2][2];

  Complex det =
      a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h;

  if (abs(det) < 1e-10) { // Threshold to avoid division by zero.
    throw std::runtime_error("Matrix is singular (not invertible).");
  }

  Matrix adjugate(3, std::vector<Complex>(3));

  adjugate[0][0] = e * i - f * h;
  adjugate[0][1] = c * h - b * i;
  adjugate[0][2] = b * f - c * e;
  adjugate[1][0] = f * g - d * i;
  adjugate[1][1] = a * i - c * g;
  adjugate[1][2] = c * d - a * f;
  adjugate[2][0] = d * h - e * g;
  adjugate[2][1] = b * g - a * h;
  adjugate[2][2] = a * e - b * d;

  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      adjugate[row][col] /= det;
    }
  }

  return SU3Matrix(adjugate);
}

// Addition assignment operator
SU3Matrix &SU3Matrix::operator+=(const SU3Matrix &rhs) {
  for (int i = 0; i < su3; ++i) {
    for (int j = 0; j < su3; ++j) {
      this->elements_[i][j] += rhs.elements_[i][j];
    }
  }
  return *this;
}

// Multiplication assignment operator
SU3Matrix &SU3Matrix::operator*=(const SU3Matrix &rhs) {
  SU3Matrix result;
  result = (*this) * rhs;
  (*this) = result;
  return *this;
}

SU3Matrix &SU3Matrix::operator*=(const Complex &rhs) {
  SU3Matrix result;
  result = (*this) * rhs;
  (*this) = result;
  return *this;
}

SU3Matrix &SU3Matrix::operator/=(const Complex &rhs) {
  SU3Matrix result;
  result = (*this) / rhs;
  (*this) = result;
  return *this;
}

Complex &SU3Matrix::operator()(int row, int col) {
  if (row >= 0 && row < 3 && col >= 0 && col < 3) {
    return elements_[row][col];

  } else {
    throw out_of_range("Invalid row or column index");
  }
}

// Complex &SU3Matrix::operator()(int row, int col) const {
//   if (row >= 0 && row < 3 && col >= 0 && col < 3) {
//     return elements_[row][col];

//   } else {
//     throw out_of_range("Invalid row or column index");
//   }
// }

const Complex &SU3Matrix::operator()(int row, int col) const {
  if (row >= 0 && row < 3 && col >= 0 && col < 3) {
    return elements_[row][col];

  } else {
    throw out_of_range("Invalid row or column index");
  }
}

void SU3Matrix::print() const {
  for (const vector<Complex> &row : elements_) {
    for (const Complex &element : row) {
      cout << element << " ";
    }
    cout << endl;
  }
}

SU3Matrix SU3Matrix::conjT() const {
  Matrix result(3, vector<Complex>(3));
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      result[i][j] = std::conj(elements_[j][i]);
    }
  }
  return SU3Matrix(result);
}

Complex SU3Matrix::det() const {
  const auto &a = elements_[0][0];
  const auto &b = elements_[0][1];
  const auto &c = elements_[0][2];
  const auto &d = elements_[1][0];
  const auto &e = elements_[1][1];
  const auto &f = elements_[1][2];
  const auto &g = elements_[2][0];
  const auto &h = elements_[2][1];
  const auto &i = elements_[2][2];

  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

Complex SU3Matrix::tr() const {
  Complex trace = 0;
  for (int i = 0; i < 3; i++) {
    trace += elements_[i][i];
  }
  return trace;
}

double SU3Matrix::reTr() const {
  double realTrace = 0;
  for (int i = 0; i < 3; i++) {
    realTrace += elements_[i][i].real();
  }
  return realTrace;
}
// USE OF
// EIGEN------------------------------------------------------------------
// SU3Matrix SU3Matrix::eigenvectors() const {
//   Eigen::Matrix3cd eigenMatrix;
//   for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//       eigenMatrix(i, j) = this->operator()(i, j);
//     }
//   }

//   // Step 2: Compute the matrix square root using Eigen
//   Eigen::ComplexEigenSolver<Eigen::Matrix3cd> solver(eigenMatrix);
//   Eigen::Matrix3cd V = solver.eigenvectors();

//   // Step 3: Convert the result back to SU3Matrix format and return
//   Matrix EV(su3, std::vector<Complex>(su3));
//   for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//       EV[i][j] = V(i, j);
//     }
//   }
//   return SU3Matrix(EV);
// }

// SU3Matrix SU3Matrix::eigenvalues() const {
//   Eigen::Matrix3cd eigenMatrix;
//   for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//       eigenMatrix(i, j) = this->operator()(i, j);
//     }
//   }

//   // Step 2: Compute the matrix square root using Eigen
//   Eigen::ComplexEigenSolver<Eigen::Matrix3cd> solver(eigenMatrix);
//   Eigen::Matrix3cd D = solver.eigenvalues().asDiagonal();

//   // Step 3: Convert the result back to SU3Matrix format and return
//   Matrix Diag(su3, std::vector<Complex>(su3));
//   for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//       Diag[i][j] = D(i, j);
//     }
//   }
//   return SU3Matrix(Diag);
// }

// SU3Matrix SU3Matrix::matrixSqrt() const {
//   // Step 1: Convert this SU3Matrix to Eigen::Matrix3cd
//   Eigen::Matrix3cd eigenMatrix;
//   for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//       eigenMatrix(i, j) = this->operator()(i, j);
//     }
//   }

//   // Step 2: Compute the matrix square root using Eigen
//   Eigen::ComplexEigenSolver<Eigen::Matrix3cd> solver(eigenMatrix);
//   Eigen::Matrix3cd D = solver.eigenvalues().asDiagonal();
//   Eigen::Matrix3cd V = solver.eigenvectors();
//   for (int i = 0; i < D.rows(); ++i) {
//     D(i, i) = std::sqrt(D(i, i));
//   }
//   Eigen::Matrix3cd sqrtEigenMatrix = V * D * V.inverse();

//   // Step 3: Convert the result back to SU3Matrix format and return
//   Matrix sqrtElements(su3, std::vector<Complex>(su3));
//   for (int i = 0; i < 3; i++) {
//     for (int j = 0; j < 3; j++) {
//       sqrtElements[i][j] = sqrtEigenMatrix(i, j);
//     }
//   }
//   return SU3Matrix(sqrtElements);
// }
//-------------------------------------------------------------------------------

void SU3Matrix::unitarize() {
  // project into SU(3) BONATI :
  // https://github.com/coppolachan/RHMC-on-GPU/blob/master/lib/Action/su3.cc#L69

  double norm = 0.0;

  for (int i = 0; i < 3; ++i) {
    norm += std::real(elements_[0][i]) * std::real(elements_[0][i]) +
            std::imag(elements_[0][i]) * std::imag(elements_[0][i]);
  }
  norm = 1.0 / std::sqrt(norm);

  for (int i = 0; i < 3; ++i) {
    elements_[0][i] *= norm;
  }

  std::complex<double> prod(0.0, 0.0);
  for (int i = 0; i < 3; ++i) {
    prod += std::conj(elements_[0][i]) * elements_[1][i];
  }

  for (int i = 0; i < 3; ++i) {
    elements_[1][i] -= prod * elements_[0][i];
  }

  norm = 0.0;
  for (int i = 0; i < 3; ++i) {
    norm += std::real(elements_[1][i]) * std::real(elements_[1][i]) +
            std::imag(elements_[1][i]) * std::imag(elements_[1][i]);
  }

  norm = 1.0 / std::sqrt(norm);
  for (int i = 0; i < 3; ++i) {
    elements_[1][i] *= norm;
  }

  prod = elements_[0][1] * elements_[1][2] - elements_[0][2] * elements_[1][1];
  elements_[2][0] = std::conj(prod);

  prod = elements_[0][2] * elements_[1][0] - elements_[0][0] * elements_[1][2];
  elements_[2][1] = std::conj(prod);

  prod = elements_[0][0] * elements_[1][1] - elements_[0][1] * elements_[1][0];
  elements_[2][2] = std::conj(prod);
}

void SU3Matrix::gramSchmidtQR() {

  SU3Matrix &Q = *this;
  SU3Matrix R;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < i; ++j) {
      subtract_projection(Q, i, Q, j);
    }
    normalize(Q, i);
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      R(i, j) = dot(*this, j, Q, i);
    }
  }

  *this = Q * R;
}

complex<double> SU3Matrix::dot(const SU3Matrix &A, int col1, const SU3Matrix &B,
                               int col2) {
  complex<double> sum = 0;
  for (int i = 0; i < 3; ++i) {
    sum += A(i, col1) * B(i, col2);
  }
  return sum;
}

void SU3Matrix::subtract_projection(SU3Matrix &A, int col, const SU3Matrix &Q,
                                    int q_col) {
  complex<double> scalar = dot(A, col, Q, q_col);
  for (int i = 0; i < 3; ++i) {
    A(i, col) -= scalar * Q(i, q_col);
  }
}

void SU3Matrix::normalize(SU3Matrix &A, int col) {
  complex<double> norm = sqrt(dot(A, col, A, col));
  for (int i = 0; i < 3; ++i) {
    A(i, col) /= norm;
  }
}

// -------------end of class functions
SU3Matrix IdentityMatrix() {
  SU3Matrix identity;
  for (int i = 0; i < su3; i++) {
    identity(i, i) = Complex(1, 0);
  }
  return identity;
}

void SU3Matrix::zeros() {
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      elements_[i][j] = Complex(0.0, 0.0);
    }
  }
}

SU3Matrix LuscherExp(SU3Matrix &W) {

  // luscher exponentiation, see appendix A of
  // https://arxiv.org/pdf/hep-lat/0409106.pdf

  SU3Matrix U1;
  SU3Matrix U2;
  SU3Matrix U3;
  SU3Matrix Id = IdentityMatrix();
  const Complex constx = Complex(0.333333, 0.0);

  Complex x1 = constx * (W(0, 0) - W(1, 1));
  Complex x2 = constx * (W(0, 0) - W(2, 2));
  Complex x3 = constx * (W(1, 1) - W(2, 2));

  SU3Matrix::Matrix elY1 = {
      {{{x1, W(0, 1), 0}}, {{W(1, 0), -x1, 0}}, {{0, 0, 0}}}};
  SU3Matrix::Matrix elY2 = {
      {{{x2, 0, W(0, 2)}}, {{0, 0, 0}}, {{W(2, 0), 0, -x2}}}};
  SU3Matrix::Matrix elY3 = {
      {{{0, 0, 0}}, {{0, x3, W(1, 2)}}, {{0, W(2, 1), -x3}}}};

  SU3Matrix Y1(elY1);
  SU3Matrix Y2(elY2);
  SU3Matrix Y3(elY3);

  Complex const1 = Complex(0.25, 0);
  Complex const2 = Complex(0.5, 0);

  U1 = (Id + Y1 * const1) * (Id - Y1 * const1).inv();
  U2 = (Id + Y2 * const1) * (Id - Y2 * const1).inv();
  U3 = (Id + Y3 * const2) * (Id - Y3 * const2).inv();

  SU3Matrix expM = U1 * U2 * U3 * U2 * U1;

  return expM;
}

SU3Matrix su3_generator() {

  SU2Matrix r = su2_generator();
  SU2Matrix s = su2_generator();
  SU2Matrix t = su2_generator();

  SU3Matrix::Matrix elementsR{
      {{{r(0, 0), r(0, 1), 0}}, {{r(1, 0), r(1, 1), 0}}, {{0, 0, 1}}}};

  SU3Matrix::Matrix elementsS{
      {{{s(0, 0), 0, s(0, 1)}}, {{0, 1, 0}}, {{s(1, 0), 0, s(1, 1)}}}};

  SU3Matrix::Matrix elementsT{
      {{{1, 0, 0}}, {{0, t(0, 0), t(0, 1)}}, {{0, t(1, 0), t(1, 1)}}}};

  SU3Matrix R(elementsR);
  SU3Matrix S(elementsS);
  SU3Matrix T(elementsT);

  SU3Matrix su3final;
  SU3Matrix su3temp;

  int tempInt = uniform_int_(0, 1);

  if (tempInt == 0) {
    su3final = R * S;
    su3final = su3final * T;

  } else if (tempInt == 1) {
    su3temp = R * S;
    su3temp *= T;
    su3final = su3temp.conjT();
  }
  return su3final;
}

// // // // // TESTED AND WORKS FINE
// int main() {

//   for (int i = 0; i < 1; i++) {
//     SU3Matrix prova = su3_generator();
//     prova.print();
//     prova.gramSchmidtQR();
//     prova.print();
//   }
// }
