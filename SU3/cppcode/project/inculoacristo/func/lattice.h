#ifndef LATTICE_H
#define LATTICE_H

#include <vector>
#include <array>
#include <complex>
#include <random>
#include <ctime>
#include <iostream>
#include "SU3Matrix.h"
#include "const.h"

class Lattice {
    public:
        using DirectionsSU3 =array<SU3Matrix, 4>;
        using LatticeSU3 = array<array<array<array<array<SU3Matrix, dir>,Nt>, Ns>, Ns>, Ns>;

        Lattice();
        ~Lattice();

        // SU3Matrix& getMatrix(int x, int y, int z, int t, int dir);
        // const SU3Matrix& getMatrix(int x, int y, int z, int t, int dir) const;

        // Overloaded operator[] to access elements
        // array<array<array<array<array<SU3Matrix, 4>,Nt>, Ns>, Ns>, Ns>& operator[](size_t index);
        // const array<array<array<array<array<SU3Matrix, 4>,Nt>, Ns>, Ns>, Ns>& operator[](size_t index) const;
    
    private:
        LatticeSU3 U;
};

#endif
