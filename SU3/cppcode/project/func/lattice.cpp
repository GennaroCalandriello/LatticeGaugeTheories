#include <iostream>
#include "lattice.h"
#include "SU3Matrix.h"
#include "const.h"

// execute with command: g++ -g3 -Wall lattice.cpp SU3Matrix.cpp distributions.cpp su2.cpp -o lattice.exe

using namespace std;

Lattice::Lattice() : U() {

}
// SU3Matrix::SU3Matrix(const Matrix& elements) : elements_(elements) {}
Lattice::~Lattice() {

}

SU3Matrix& Lattice::operator()(int x, int y, int z, int t, int dir) {
    return U[x][y][z][t][dir];
}
const SU3Matrix& Lattice::operator()(int x, int y, int z, int t, int dir) const {
    return U[x][y][z][t][dir];
}

// fill U with su3 matrices:
Lattice fill() {
    Lattice U;
    for (int x = 0; x < Ns; x++) {
        for (int y = 0; y < Ns; y++) {
            for (int z = 0; z < Ns; z++) {
                for (int t = 0; t < Nt; t++) {
                    for (int dir = 0; dir < 4; dir++) {
                        U(x, y, z, t, dir)= su3_generator();
                    }
                }
            }
        }
    }
    return U;
}
void printConfiguration(Lattice U){
    for (int x = 0; x < su3;x++){
        for (int y = 0; y < su3; y++){
            for (int z = 0; z < su3; z++){
                for (int t = 0; t < su3; t++){
                    for (int dir = 0; dir < 4; dir++){
                        U(x, y, z, t, dir).print();
                    }
                }
            }
        }
    }
}
int main(){
    Lattice U;
    U = fill();
    
    std::cout << "Hello, Diomerda!" << std::endl;
    printConfiguration(U);
    std::cout << U(0,0,0,0,0).print() << std::endl;
    std::cout << U(0, 1, 1, 3, 0).det() << "queso è il fottuto determinante" << std::endl;
    return 0;
}

