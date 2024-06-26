#include "su2.h"
#include "const.h"
#include "distributions.h"
#include "paulimatrices.h"
#include <complex>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using Complex = complex<double>;

// Element access
Complex &SU2Matrix::operator()(int row, int col) {
    return elements_[row * 2 + col];
}

const Complex &SU2Matrix::operator()(int row, int col) const {
    return elements_[row * 2 + col];
}

// Generate a random SU(2) matrix
SU2Matrix su2_generator() {
    SU2Matrix su2_matrix;

    double r0 = uniform_(-0.5, 0.499);
    int sign = (r0 >= 0) ? 1 : -1;
    double x0 = sign * std::sqrt(1 - std::pow(epsilon, 2));

    std::vector<double> r(3, 0);
    std::vector<double> r_norm(3, 0);
    SU2Matrix I = IdentityMat();

    for (int i = 0; i < 3; i++) {
        r[i] = uniform_(0, 0.999) - 0.5;
    }

    double norm = std::sqrt(std::pow(r[0], 2) + std::pow(r[1], 2) + std::pow(r[2], 2));

    for (int i = 0; i < 3; i++) {
        r_norm[i] = epsilon * r[i] / norm;
    }

    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 2; b++) {
            Complex temp = x0 * I(a, b) + Complex(0, 1) * r_norm[0] * s_x[a][b] +
                           Complex(0, 1) * r_norm[1] * s_y[a][b] +
                           Complex(0, 1) * r_norm[2] * s_z[a][b];

            su2_matrix(a, b) = temp;
        }
    }

    return su2_matrix;
}

// Print the matrix
void SU2Matrix::print() const {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << operator()(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Identity matrix
SU2Matrix IdentityMat() {
    SU2Matrix identity;
    identity(0, 0) = Complex(1, 0);
    identity(0, 1) = Complex(0, 0);
    identity(1, 0) = Complex(0, 0);
    identity(1, 1) = Complex(1, 0);
    return identity;
}

// da qui

// Addition of two SU2 matrices
SU2Matrix SU2Matrix::operator+(const SU2Matrix &other) const {
    SU2Matrix result;
    for (int i = 0; i < 4; ++i) {
        result.elements_[i] = elements_[i] + other.elements_[i];
    }
    return result;
}

// Subtraction of two SU2 matrices
SU2Matrix SU2Matrix::operator-(const SU2Matrix &other) const {
    SU2Matrix result;
    for (int i = 0; i < 4; ++i) {
        result.elements_[i] = elements_[i] - other.elements_[i];
    }
    return result;
}

// Multiplication of two SU2 matrices
SU2Matrix SU2Matrix::operator*(const SU2Matrix &other) const {
    SU2Matrix result;
    result(0, 0) = elements_[0] * other(0, 0) + elements_[1] * other(1, 0);
    result(0, 1) = elements_[0] * other(0, 1) + elements_[1] * other(1, 1);
    result(1, 0) = elements_[2] * other(0, 0) + elements_[3] * other(1, 0);
    result(1, 1) = elements_[2] * other(0, 1) + elements_[3] * other(1, 1);
    return result;
}

// Scalar multiplication
SU2Matrix SU2Matrix::operator*(const Complex &rhs) const {
    SU2Matrix result;
    for (int i = 0; i < 4; ++i) {
        result.elements_[i] = elements_[i] * rhs;
    }
    return result;
}

SU2Matrix SU2Matrix::operator*(const double &rhs) const {
    SU2Matrix result;
    for (int i = 0; i < 4; ++i) {
        result.elements_[i] = elements_[i] * rhs;
    }
    return result;
}

// Addition assignment operator
SU2Matrix &SU2Matrix::operator+=(const SU2Matrix &rhs) {
    for (int i = 0; i < 4; ++i) {
        elements_[i] += rhs.elements_[i];
    }
    return *this;
}

// Multiplication assignment operator
SU2Matrix &SU2Matrix::operator*=(const SU2Matrix &rhs) {
    *this = *this * rhs;
    return *this;
}

// Scalar division
SU2Matrix &SU2Matrix::operator/=(const Complex &rhs) {
    for (int i = 0; i < 4; ++i) {
        elements_[i] /= rhs;
    }
    return *this;
}

SU2Matrix &SU2Matrix::operator/=(const double &rhs) {
    for (int i = 0; i < 4; ++i) {
        elements_[i] /= rhs;
    }
    return *this;
}

// Conjugate transpose
SU2Matrix SU2Matrix::conjT() const {
    SU2Matrix result;
    result(0, 0) = std::conj(operator()(0, 0));
    result(0, 1) = std::conj(operator()(1, 0));
    result(1, 0) = std::conj(operator()(0, 1));
    result(1, 1) = std::conj(operator()(1, 1));
    return result;
}

// Determinant
Complex SU2Matrix::det() const {
    return operator()(0, 0) * operator()(1, 1) - operator()(0, 1) * operator()(1, 0);
}

// Trace
Complex SU2Matrix::tr() const {
    return operator()(0, 0) + operator()(1, 1);
}

// Real part of the trace
double SU2Matrix::reTr() const {
    return std::real(operator()(0, 0) + operator()(1, 1));
}
// int main(){
//     SU2Matrix su2_matrix = su2_generator();
//     SU2Matrix identity = IdentityMat();
//     su2_matrix.print();
//     cout << "Sum of the two matrices:" << endl;
//     sum.print();
//     return 0;
// }