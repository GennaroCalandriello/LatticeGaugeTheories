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

// to compile all the files included the command is: g++ -g3 -Wall SU3Matrix.cpp
// su2.cpp distributions.cpp -o SU3Matrix.exe

// implement parametrized constructor
SU3Matrix::SU3Matrix(const Matrix &elements) : elements_(elements) {}

// implement destructor
SU3Matrix::~SU3Matrix() {}

SU3Matrix SU3Matrix::operator+(const SU3Matrix &other) const {
  Matrix result;
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      result[i][j] = elements_[i][j] + other.elements_[i][j];
    }
  }
  return *this;
}

SU3Matrix SU3Matrix::operator-(const SU3Matrix &other) const {
  Matrix result;
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++)
      result[i][j] = elements_[i][j] - other.elements_[i][j];
  }
  return *this;
}

SU3Matrix SU3Matrix::operator*(const SU3Matrix &other) const {
  Matrix result;
  for (int i = 0; i < su3; i++) {
    for (int j = 0; j < su3; j++) {
      for (int k = 0; k < su3; k++) {
        result[i][j] += elements_[i][k] * other.elements_[k][j];
      }
    }
  }
  return result;
}

// Addition assignment operator
SU3Matrix &SU3Matrix::operator+=(const SU3Matrix &rhs) {
  for (int i = 0; i < su3; ++i) {
    for (int j = 0; j < su3; ++j) {
      elements_[i][j] += rhs.elements_[i][j];
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

// QSMatrix<T>& QSMatrix<T>::operator*=(const QSMatrix<T>& rhs) {
//   QSMatrix result = (*this) * rhs;
//   (*this) = result;
//   return *this;

Complex &SU3Matrix::operator()(int row, int col) {
  if (row >= 0 && row < 3 && col >= 0 && col < 3) {
    return elements_[row][col];

  } else {
    throw out_of_range("Invalid row or column index");
  }
}

void SU3Matrix::print() const {
  for (const auto &row : elements_) {
    for (const auto &element : row) {
      cout << element << " ";
    }
    cout << endl;
  }
}

SU3Matrix SU3Matrix::conjT() const {
  Matrix result = {};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result[i][j] = std::conj(elements_[j][i]);
    }
  }
  return *this;
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

SU3Matrix IdentityMatrix() {
  SU3Matrix identity;
  for (int i = 0; i < su3; i++) {
    identity(i, i) = Complex(1, 0);
  }
  return identity;
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

//   for (int i = 0; i < 100; i++) {
//     SU3Matrix prova = su3_generator();
//     cout << "fineeee" << endl;
//     cout << prova.reTr() << endl;
//   }
// }
