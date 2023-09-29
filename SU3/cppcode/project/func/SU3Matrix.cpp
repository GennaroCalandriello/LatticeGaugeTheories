#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <ctime>


#include "SU3Matrix.h"
#include "su2.h"
#include "distributions.h"
#include "const.h"

using namespace std;
using Complex = complex<double>;

// to compile all the files included the command is: g++ -g3 -Wall SU3Matrix.cpp su2.cpp distributions.cpp -o SU3Matrix.exe

// implement parametrized constructor
SU3Matrix::SU3Matrix(const Matrix& elements) : elements_(elements) {}

// implement destructor
SU3Matrix::~SU3Matrix() {}


SU3Matrix SU3Matrix::operator+(const SU3Matrix& other) const{
    Matrix result;
    for (int i = 0; i < su3; i++){
        for (int j = 0; j < su3; j++){
            result[i][j] = elements_[i][j] + other.elements_[i][j];
        }
    }
    return *this;
}

SU3Matrix SU3Matrix::operator-(const SU3Matrix& other) const{
    Matrix result;
    for (int i = 0; i < su3; i++)
    {
        for (int j = 0; j < su3; j++)
        result[i][j] = elements_[i][j] - other.elements_[i][j];
    }
    return *this;
}

SU3Matrix SU3Matrix::operator*(const SU3Matrix& other) const{
    Matrix result;
    for (int i = 0; i < su3; i++)
    {
        for (int j = 0; j < su3; j++)
        {
            for (int k = 0; k < su3; k++)
            {
                result[i][j] += elements_[i][k] * other.elements_[k][j];
            }
                
        }
    }
    return *this;
}

// Addition assignment operator
SU3Matrix& SU3Matrix:: operator+=(const SU3Matrix& rhs) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            elements_[i][j] += rhs.elements_[i][j];
        }
    }
    return *this;
}

// Multiplication assignment operator
SU3Matrix& SU3Matrix::operator*=(const SU3Matrix& rhs) {
    Matrix result = {};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                result[i][j] += elements_[i][k] * rhs.elements_[k][j];
            }
        }
    }
    elements_ = result;
    return *this;
}

Complex SU3Matrix::operator()(int row, int col) const {
    if (row >= 0 && row < 3 && col >= 0 && col < 3) {
        return elements_[row][col];

    } else {
        throw out_of_range("Invalid row or column index");
    }
}
    

void SU3Matrix::print() const {
    for (const auto& row : elements_){
        for (const auto& element : row){
            cout << element << " ";
        }
        cout << endl;
    }
}

SU3Matrix SU3Matrix::conjT() const {
    Matrix result = {};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result[i][j] = std::conj(elements_[j][i]);
        }
    }
    return *this;
}

Complex SU3Matrix::det() const {
        const auto& a = elements_[0][0];
        const auto& b = elements_[0][1];
        const auto& c = elements_[0][2];
        const auto& d = elements_[1][0];
        const auto& e = elements_[1][1];
        const auto& f = elements_[1][2];
        const auto& g = elements_[2][0];
        const auto& h = elements_[2][1];
        const auto& i = elements_[2][2];

        return a * (e*i - f*h) - b * (d*i - f*g) + c * (d*h - e*g);
}

 

SU3Matrix su3_generator() {

    ComplexMatrix2x2 r = su2_matrix();
    ComplexMatrix2x2 s = su2_matrix();
    ComplexMatrix2x2 t = su2_matrix();

    SU3Matrix::Matrix elementsR{{
        {{r[0][0], r[0][1], 0}},
        {{r[1][0], r[1][1], 0}},
        {{0, 0, 1}}
        }};

    SU3Matrix::Matrix elementsS{{
        {{s[0][0], 0, s[0][1]}},
        {{0, 1, 0}},
        {{s[1][0], 0, s[1][1]}}
        }};

    SU3Matrix::Matrix elementsT{{
        {{1, 0, 0}},
        {{0, t[0][0], t[0][1]}},
        {{0, t[1][0], t[1][1]}}
        }};

    SU3Matrix R(elementsR);
    SU3Matrix S(elementsS);
    SU3Matrix T(elementsT);
    

    int tempInt = uniform_int_(0, 2);

    if (tempInt == 0)
    {
        SU3Matrix su3temp = R * S;
        // R.print();
        // S.print();  
        // su3temp.print();  
        SU3Matrix su3final = su3temp * T; 

        return su3final;

    } else if(tempInt != 0)
    {
        SU3Matrix su3temp = R*S;
        SU3Matrix su3final = su3temp*T;

        return su3final.conjT();
    }
    // Add a default return statement
}

// int main() {

//     SU3Matrix prova = su3_generator();
//     complex<double> det = prova.det();
//     cout << prova(0, 1) << endl;
//     cout << det << endl;

// }
