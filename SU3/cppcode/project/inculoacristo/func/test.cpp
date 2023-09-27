#include "su3.h"
#include <vector>
#include <iostream>
#include <string>
#include "lattice.h"

using namespace std;

int main(int argc, char **argv) {
    SU3Matrix<complex<double>> mat1;
    cout << mat1(0, 1) << endl;
    
}