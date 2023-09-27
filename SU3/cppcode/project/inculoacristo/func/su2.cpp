#include <complex>
#include <random>
#include <iostream>
#include <array>
#include <complex>

#include "paulimatrices.h"
#include "su2.h"
#include "distributions.cpp"

const double epsilon = 0.1;
using namespace std;

ComplexMatrix2x2 su2_matrix(){

    srand (clock());    

    ComplexMatrix2x2 SU2Matrix;

    double r0 = uniform_();
    double x0 = copysign(sqrt(1 - pow(epsilon, 2)), r0);
    
    array<double, 3> r;
    for (int i = 0; i < 3; i++)
    {
        r[i] = uniform_()-0.5;
    }

    double norm = sqrt(pow(r[0], 2) + pow(r[1], 2) + pow(r[2], 2));

    for (int i = 0; i < 3; i++)
    {
        r[i] = epsilon * r[i] / norm;
    }

    for (int a = 0; a < 2; a++)
    {
        for (int b = 0; b < 2; b++)
        {
            complex<double> temp = x0 * Id[a][b] + 1i * r[0] * s_x[a][b] + 1i * r[1] * s_y[a][b] + 1i * r[2] * s_z[a][b];

            // double realpart = temp.real();
            // double imagpart = temp.imag();

            SU2Matrix[a][b] = temp;
            // SU2Matrix[a][b] = x0 * Id[a][b] + 1i * r[0] * s_x[a][b] + 1i * r[1] * s_y[a][b] + 1i * r[2] * s_z[a][b];
        }
    }
    
    return SU2Matrix;
}
