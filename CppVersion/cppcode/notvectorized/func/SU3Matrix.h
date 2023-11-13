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
  SU3Matrix operator/(const Complex &other) const; // scalar division
  SU3Matrix operator*(const Complex &rhs) const;   // scalar multiplication
  SU3Matrix operator+(const Complex &other) const; // scalar addition
  SU3Matrix operator-(const Complex &other) const; // scalar subtraction
  SU3Matrix &operator*=(const Complex &rhs);
  SU3Matrix &operator/=(const Complex &rhs);
  SU3Matrix &operator+=(const SU3Matrix &rhs);
  SU3Matrix &operator*=(const SU3Matrix &rhs);

  SU3Matrix matrixSqrt() const;
  SU3Matrix eigenvectors() const;
  SU3Matrix eigenvalues() const;
  SU3Matrix inv();

  Complex &operator()(int row, int col);
  // Complex &operator()(int row, int col) const;
  const Complex &operator()(int row, int col) const;

  void zeros();

  void print() const;

  SU3Matrix conjT() const;
  Complex det() const;
  Complex tr() const;
  double reTr() const;
  void unitarize();
  void gramSchmidtQR();
  complex<double> dot(const SU3Matrix &A, int col1, const SU3Matrix &B,
                      int col2);
  void subtract_projection(SU3Matrix &A, int col, const SU3Matrix &Q,
                           int q_col);
  void normalize(SU3Matrix &A, int col);

private:
  Matrix elements_;
};

SU3Matrix su3_generator();
SU3Matrix IdentityMatrix();
SU3Matrix LuscherExp(SU3Matrix &W);

#endif // SU3MATRIX_H
