#include <iostream>
#include <ctime>
#include <complex>
#include <random>
#include "../func/lattice.h"
#include "../func/su3.h"
#include "../func/matrices.cpp"
#include "../func/algebra.h"

SU3 staple(Lattice& U, int x, int y, int z, int t, int mu){

    SU3 W1;
    SU3 W2;
    SU3 Wf;

    vector<int> a_mu(4, 0);
    a_mu[mu] = 1;
    vector<int> pmu = index(x, y, z, t, mu, 1, a_mu, "f");
    vector<int> mmu = index(x, y, z, t, mu, 1, a_mu, "b");
    
    for (int nu = 0; nu < dir; nu++) 
    {
        vector<int> a_nu(4, 0);
        a_nu[nu] = 1;
        vector<int> pnu = index(x, y, z, t, nu, 1, a_nu, "f");
        vector<int> mnu = index(x, y, z, t, nu, 1, a_nu, "b");

        if (nu == mu)
        {
            continue;
        }
        vector<int> pmu_m_nu = index(pmu[0], pmu[1], pmu[2], pmu[3], nu, 1, a_nu, "b");


        W1 = matmul(
            U[pmu[0]][pmu[1]][pmu[2]][pmu[3]][nu], 
            matmul(
                U[pnu[0]][pnu[1]][pnu[2]][pnu[3]][mu], conjT(U[x][y][z][t][nu])));
                
        W2 = matmul(
            conjT(U[pmu_m_nu[0]][pmu_m_nu[1]][pmu_m_nu[2]][p_mu_m_nu[3]][nu]),
            matmul(
                conjT(U[mnu[0]][mnu[1]][mnu[2]][mnu[3]][mu]), U[mnu[0]][mnu[1]][mnu[2]][mnu[3]][nu]));

        add(Wf, W1, W2); // should perform Wf += W1+W2
    }
    
    return Wf;
}

vector<int> index(const int x,const  int y, const int z,const  int t, int dir, int l, const vector<int> a_dir, const string& direction)
{
    if (direction =="f")
    {
        int xp = (x + l * a_dir[dir]) % Ns;
        int yp = (y + l * a_dir[dir]) % Ns;
        int zp = (y + l * a_dir[dir]) % Ns;
        int tp = (t + l * a_dir[dir]) % Nt;

        return {xp, yp, zp, tp};
    }

    if (direction =="b")
    {
        int xm = (x - l * a_dir[dir]) % Ns;
        int ym = (y - l * a_dir[dir]) % Ns;
        int zm = (y - l * a_dir[dir]) % Ns;
        int tm = (t - l * a_dir[dir]) % Nt;

        return {xm, ym, zm, tm};
    }

}

complex<double> Wilson(Lattice& U)

{
    complex<double> S = 0;
    for (int x = 0; x < Ns; x++)
    {
        for (int y = 0; y < Ns; y++)
        {
            for (int z = 0; z < Ns; z++)
            {
                for (int t = 0; t < Nt; t++)
                {
                    SU3 temp;
                    for (int mu = 0; mu < dir; mu++)
                    {
                        vector<int> a_mu(4, 0);
                        a_mu[mu] = 1;

                        for (int nu = 0; nu < dir; nu++)
                        {
                            vector<int> a_nu(4, 0);
                            a_nu[nu] = 1;

                                                }
                            
                    }
                }
            }
        }
    }
        
    return S;
}