#ifndef LATTICE_H
#define LATTICE_H

#include "SU3Matrix.h"
#include "const.h"
#include <complex>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using Complex = complex<double>;

class Lattice {
public:
  using LatticeSU3 = std::vector<
      std::vector<std::vector<std::vector<std::vector<SU3Matrix>>>>>;

  Lattice()
      : U(Ns, std::vector<std::vector<std::vector<std::vector<SU3Matrix>>>>(
                  Ns, std::vector<std::vector<std::vector<SU3Matrix>>>(
                          Ns, std::vector<std::vector<SU3Matrix>>(
                                  Nt, std::vector<SU3Matrix>(dir))))){};

  ~Lattice() = default;

  SU3Matrix &operator()(int x, int y, int z, int t, int mu) {
    return U[x][y][z][t][mu];
  }

private:
  LatticeSU3 U;
};

Lattice fill();
void printConfiguration(Lattice U);

#endif

//  la sintassi "Lattice::LatticeSU3& Lattice::operator[](size_t index)"
//  definisce l'operatore di indicizzazione operator[] per la classe Lattice.
//  Vediamo ogni parte di questa dichiarazione:

// Lattice::LatticeSU3&

// Lattice::: Questa parte indica che stiamo definendo qualcosa all'interno
// della classe Lattice. È il modo in cui C++ sa che questa funzione o metodo
// appartiene a quella classe.
// LatticeSU3&: Questa parte è il tipo di ritorno del metodo. Qui stiamo dicendo
// che il metodo ritorna un riferimento (&) a un oggetto di tipo LatticeSU3 (che
// è un tipo definito all'interno della classe Lattice).
// Lattice::operator[](size_t index)

// Lattice::: Come prima, questo indica che stiamo definendo un metodo che
// appartiene alla classe Lattice. operator[]: Questa è la sintassi speciale per
// definire l'operatore di indicizzazione. Si tratta di una funzione
//  membro che si può chiamare usando la sintassi degli array, come
//  oggetto[indice].
// (size_t index): Questa è la lista dei parametri del metodo. In questo caso,
// il metodo accetta un solo parametro,
//  di tipo size_t (un tipo di dato intero senza segno), che rappresenta
//  l'indice da accedere.