#ifndef SU2_MATRIX_H
#define SU2_MATRIX_H

#include "const.h"
#include <complex>
#include <vector>

using namespace std;
using Complex = complex<double>;

class SU2Matrix {
public:
  using Matrix = vector<vector<Complex>>;

  // default constructor
  SU2Matrix() : elements_(su2, vector<Complex>(su2, Complex(0.0, 0.0))){};

  // Constructor that takes a Matrix as a parameter
  SU2Matrix(const Matrix &elements) : elements_(elements) {}

  // destructor
  ~SU2Matrix() {}

  SU2Matrix operator+(const SU2Matrix &other) const;
  SU2Matrix operator-(const SU2Matrix &other) const;
  SU2Matrix operator*(const SU2Matrix &other) const;
  SU2Matrix operator*(const Complex &rhs) const;
  SU2Matrix operator*(const double &rhs) const;
  SU2Matrix &operator+=(const SU2Matrix &rhs);
  SU2Matrix &operator*=(const SU2Matrix &rhs);
  SU2Matrix &operator/=(const Complex &rhs); // scalar division
  SU2Matrix &operator/=(const double &rhs);  // scalar division
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
