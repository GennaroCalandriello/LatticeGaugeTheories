// Include guards to prevent double inclusion
#ifndef SU2_MATRIX_H
#define SU2_MATRIX_H

// Necessary includes
#include "const.h"
#include <array>
#include <complex>

using namespace std;
using Complex = complex<double>;

class SU2Matrix {
public:
  using Matrix = array<array<Complex, 2>, 2>;

  // default constructor
  SU2Matrix() {
    for (int i = 0; i < su2; i++) {
      for (int j = 0; j < su2; j++) {
        elements_[i][j] = Complex(0.0, 0.0);
      }
    }
  };

  // Constructor that takes a Matrix as a parameter (PARAMETRIC CONSTRUCTOR)
  SU2Matrix(Matrix &elements);
  // destructor
  ~SU2Matrix();

  SU2Matrix operator+(const SU2Matrix &other) const;
  SU2Matrix operator-(const SU2Matrix &other) const;
  SU2Matrix operator*(const SU2Matrix &other) const;
  SU2Matrix &operator+=(const SU2Matrix &rhs);
  SU2Matrix &operator*=(const SU2Matrix &rhs);
  Complex &operator()(int row, int col);

  void print() const;
  SU2Matrix conjT() const;
  Complex det() const;
  Complex tr() const;
  double reTr() const;

private:
  Matrix elements_;
};

SU2Matrix su2_generator();
SU2Matrix IdentityMat();

#endif // SU2_MATRIX_H