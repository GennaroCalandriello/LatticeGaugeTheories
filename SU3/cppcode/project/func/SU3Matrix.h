#ifndef SU3MATRIX_H
#define SU3MATRIX_H

#include <complex>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "const.h"

using namespace std;
using Complex = complex<double>;

class SU3Matrix {
public:
  using Matrix = vector<vector<Complex>>;

  // default constructor
  SU3Matrix() : elements_(su3, vector<Complex>(su3, Complex(0.0, 0.0))) {}

  // Constructor that takes a Matrix as a parameter (PARAMETRIC CONSTRUCTOR)
  // in C++ it is possible to declare multiple constructors
  SU3Matrix(Matrix &elements) : elements_(elements) {}

  // destructor
  ~SU3Matrix() {}

  SU3Matrix operator+(const SU3Matrix &other) const;
  SU3Matrix operator-(const SU3Matrix &other) const;
  SU3Matrix operator*(const SU3Matrix &other) const;
  SU3Matrix &operator+=(const SU3Matrix &rhs);
  SU3Matrix &operator*=(const SU3Matrix &rhs);
  Complex &operator()(int row, int col);
  void zeros();

  void print() const;

  SU3Matrix conjT() const;
  Complex det() const;
  Complex tr() const;
  double reTr() const;

private:
  Matrix elements_;
};

SU3Matrix su3_generator();
SU3Matrix IdentityMatrix();

#endif // SU3MATRIX_H
