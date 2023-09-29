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
        // LatticeSU3& operator[](size_t index);

        SU3Matrix& operator()(int x, int y, int z, int t, int dir);
        const SU3Matrix& operator()(int x, int y, int z, int t, int dir) const;
       
    
    private:
        LatticeSU3 U;
};
Lattice fill();
void printConfiguration(Lattice U);

#endif


//  la sintassi "Lattice::LatticeSU3& Lattice::operator[](size_t index)" definisce l'operatore di indicizzazione operator[] per la classe Lattice. Vediamo ogni parte di questa dichiarazione:

// Lattice::LatticeSU3&

// Lattice::: Questa parte indica che stiamo definendo qualcosa all'interno della classe Lattice. È il modo in cui C++ sa che questa funzione o metodo 
// appartiene a quella classe.
// LatticeSU3&: Questa parte è il tipo di ritorno del metodo. Qui stiamo dicendo che il metodo ritorna un riferimento (&) a un oggetto di tipo 
// LatticeSU3 (che è un tipo definito all'interno della classe Lattice).
// Lattice::operator[](size_t index)

// Lattice::: Come prima, questo indica che stiamo definendo un metodo che appartiene alla classe Lattice.
// operator[]: Questa è la sintassi speciale per definire l'operatore di indicizzazione. Si tratta di una funzione
//  membro che si può chiamare usando la sintassi degli array, come oggetto[indice].
// (size_t index): Questa è la lista dei parametri del metodo. In questo caso, il metodo accetta un solo parametro,
//  di tipo size_t (un tipo di dato intero senza segno), che rappresenta l'indice da accedere.