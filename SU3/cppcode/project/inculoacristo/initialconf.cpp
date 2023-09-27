#include <iostream>
#include <ctime>
#include <complex>
#include <random>
#include "func/lattice.h"
#include "func/SU3Matrix.cpp"
#include "initialconf.h"


using namespace std;




Lattice initialize_lattice()
{
    Lattice U;
    srand (clock());


    for (int x = 0; x < Ns; x++)
    {
        for (int y = 0; y < Ns; y++)
        {
            for (int z = 0; z < Ns; z++)
            {
                for (int t = 0; t < Nt; t++)
                {
                    for (int mu = 0; mu < dir; mu++)
                    {
                        SU3Matrix su3M = su3_generator();
                        for (int a = 0; a < su3; a++)
                        {
                            for (int b = 0; b < su3; b++)
                            {
                                double realpart = su3M(a, b).real();
                                double imagpart = su3M(a, b).imag();
                                
                                U[x][y][z][t][mu][a][b] = std::complex<double>(realpart, imagpart);
                            }
                        }
                    }
                }
            }
        }
    }
    return U;
}


int main()
{
    srand (clock());
    Lattice lattice = initialize_lattice();
    SU3Matrix temp = lattice[0][0][0][0][0];
    cout << "quest est nu testt" << endl;
    cout << temp(0, 2) << endl;
    // SU3Matrix temp2 = (temp).conjT();
    
}
