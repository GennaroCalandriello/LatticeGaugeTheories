#ifndef SU2_MATRIX_H
#define SU2_MATRIX_H

#include "const.h"
#include <complex>
#include <vector>
#include <iostream>

using namespace std;
using Complex = complex<double>;

class SU2Matrix {
public:
    // Array to store elements
    Complex elements_[4];

    // Default constructor
    SU2Matrix() {
        for (int i = 0; i < 4; ++i) {
            elements_[i] = Complex(0.0, 0.0);
        }
    }

    // Constructor that takes a vector of vectors as a parameter
    // Destructor
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
    const Complex &operator()(int row, int col) const;

    void print() const;
    SU2Matrix conjT() const;
    Complex det() const;
    Complex tr() const;
    double reTr() const;
};

// Function to generate a random SU(2) matrix
SU2Matrix su2_generator();
SU2Matrix IdentityMat();

#endif // SU2_MATRIX_H
