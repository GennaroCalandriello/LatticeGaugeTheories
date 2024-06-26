// #include "Eigen/Dense"
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

// g++ -g3 -Wall SU3Matrix.cpp su2.cpp distributions.cpp -o SU3Matrix.exe
// g++ -g3 -Wall -I/eigen/Eigen SU3Matrix.cpp su2.cpp distributions.cpp -o
// SU3Matrix.exe

// implement parametrized constructor
// SU3Matrix::SU3Matrix(const Matrix &elements) : elements_(elements) {}

// implement destructor
// SU3Matrix::~SU3Matrix() {}


Complex &SU3Matrix::operator()(int row, int column) {
  return elements_[row * 3 + column];

}

const Complex &SU3Matrix::operator()(int row, int column) const {
    return elements_[row * 3 + column];
  }

  // SU3Matrix operator+(const SU3Matrix &other) const {
  //   SU3Matrix result;
  //   for (int i = 0; i < 9; i++) {
  //     result.elements[i] = elements[i] + other.elements[i];
  //   }
  //   return result;
  // }

  // SU3Matrix operator*(const SU3Matrix& other) const{
  //   SU3Matrix result;
  //   for (int i = 0; i < 3;i++){
  //     for (int j = 0; j < 3; j++){
  //       Complex sum(0.0, 0.0);
  //       for (int k = 0; k < 3; k++)
  //       {
  //         sum += operator()(i, k) * other(k, j);
  //       }
  //     }
  //   }
  //   return result;
  // }

int main(){
return 0;
}