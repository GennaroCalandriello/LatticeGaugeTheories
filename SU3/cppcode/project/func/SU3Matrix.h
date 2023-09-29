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

        //default construnctor
        SU3Matrix()  {
            // initialize elements to produce a default constructor
            for (int i = 0; i < su3;i++){
                for (int j = 0; j < su3; j++){
                    elements_[i][j] = Complex(0.0, 0.0);
                }
            }
        };

        // Constructor that takes a Matrix as a parameter (PARAMETRIC CONSTRUCTOR)
        // in C++ it is possible to declare multiple constructors
        SU3Matrix(const Matrix& elements);
        // destructor
        ~SU3Matrix();

        SU3Matrix operator+(const SU3Matrix& other) const;
        SU3Matrix operator-(const SU3Matrix& other) const;
        SU3Matrix operator*(const SU3Matrix& other) const;
        SU3Matrix& operator+=(const SU3Matrix& rhs);
        SU3Matrix& operator*=(const SU3Matrix& rhs);
        Complex& operator()(int row, int col);

        void print() const;
        SU3Matrix conjT() const;
        Complex det() const;
        Complex tr() const;
        double reTr() const;
        

    

    private: 
        Matrix elements_;
};

SU3Matrix su3_generator();
SU3Matrix Id();

#endif  // SU3MATRIX_H