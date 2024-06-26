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
    // Array to store elements
    Complex elements_[9];

    // Default constructor
    SU3Matrix() {
        for (int i = 0; i < 9; ++i) {
            elements_[i] = Complex(0.0, 0.0);
        }
    }

    // Destructor
    ~SU3Matrix() {}

    SU3Matrix operator+(const SU3Matrix& other) const;
    SU3Matrix operator-(const SU3Matrix& other) const;
    SU3Matrix operator*(const SU3Matrix& other) const;
    SU3Matrix operator/(const Complex& other) const; // scalar division
    SU3Matrix operator*(const Complex& rhs) const;   // scalar multiplication
    SU3Matrix operator+(const Complex& other) const; // scalar addition
    SU3Matrix operator-(const Complex& other) const; // scalar subtraction
    SU3Matrix& operator*=(const Complex& rhs);
    SU3Matrix& operator/=(const Complex& rhs);
    SU3Matrix& operator+=(const SU3Matrix& rhs);
    SU3Matrix& operator*=(const SU3Matrix& rhs);

    SU3Matrix matrixSqrt() const;
    SU3Matrix eigenvectors() const;
    SU3Matrix eigenvalues() const;
    SU3Matrix inv();

    Complex &operator()(int row, int col);
    const Complex &operator()(int row, int col) const;

    void zeros();

    void print() const;

    SU3Matrix conjT() const;
    Complex det() const;
    Complex tr() const;
    double reTr() const;
    void unitarize();
    void gramSchmidtQR();
    complex<double> dot(const SU3Matrix& A, int col1, const SU3Matrix& B, int col2);
    void subtract_projection(SU3Matrix& A, int col, const SU3Matrix& Q, int q_col);
    void normalize(SU3Matrix& A, int col);
};

// Function to generate a random SU(3) matrix
SU3Matrix su3_generator();
SU3Matrix IdentityMatrix();
SU3Matrix LuscherExp(SU3Matrix& W);

#endif // SU3MATRIX_H
