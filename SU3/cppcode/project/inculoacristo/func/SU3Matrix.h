#ifndef SU3MATRIX_H
#define SU3MATRIX_H

#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <ctime>

#include "const.h"



using namespace std;
using Complex = complex<double>;

class SU3Matrix{
    public:
        using Matrix = array<array<Complex, su3>, su3>;

        // construnctor
        SU3Matrix(const Matrix &elements);
        // destructor
        ~SU3Matrix();

        SU3Matrix operator+(const SU3Matrix& other) const;
        SU3Matrix operator-(const SU3Matrix& other) const;
        SU3Matrix operator*(const SU3Matrix& other) const;
        SU3Matrix& operator+=(const SU3Matrix& rhs);
        SU3Matrix& operator*=(const SU3Matrix& rhs);
        

        Complex operator()(int row, int col) const;
        void print() const;
        SU3Matrix conjT() const;
        Complex det() const;
    

    private: 
        Matrix elements_;
};

SU3Matrix su3_generator();

#endif  // SU3MATRIX_H