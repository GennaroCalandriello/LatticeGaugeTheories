from math import sqrt
from shutil import move
import numpy as np
from numba import njit, float64, complex128, int64
import time
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt

from HB_OR_SU3 import *
from functions import *
from parameters import *

from numba import config

config.FULL_TRACEBACKS = True

tau = 1  # trajectory length
Nstep = 4  # number of leapfrog steps
dtau = tau / Nstep  # 1 / Nstep  # leapfrog step dtau =0.0375 setting from reference [4]


"""references:
[1] S. Schaefer, R. Sommer, and F. Virotta, “Critical slowing down and error analysis in lattice QCD simulations,” Nucl. Phys. B, vol. 845, no. 1, pp. 93–114, 2011.   
[2] M. Lüscher, “Topology, the Wilson flow and the HMC algorithm,” PoS, vol. LATTICE2013, p. 016, 2014.
[3] M. Lüscher, “Properties and uses of the Wilson flow in lattice QCD,” JHEP, vol. 08, p. 071, 2010.
[4] R. Gupta et al "Comparison of Update Algorithms for Pure Gauge SU(3)", http://cds.cern.ch/record/186962/files/CM-P00052019.pdf
[5] Kennedy, Pendleton "ACCEPTANCES AND AUTOCORRELATIONS IN HYBRID MONTE CARLO"

alcune regole:
1. nelle funzioni se c'è U viene prima di tutti gli altri argomenti
2. ordine generale (U, P, beta, x, y, z, t, mu):


NOTA IMPORTANTE: njit(argomentooutput(argomentiinput), fastmath =True) funziona velocissimo
ma le funzioni devono essere strutturate per cacare solo 1 argomento alla volta
"""


def another_initializeLattice():
    """Initialize the lattice with random SU(3) matrices using U =exp(iQ)"""

    import scipy.linalg as la

    U = np.zeros((Ns, Ns, Ns, Nt, 4, 3, 3), dtype=np.complex128)

    def random_su3_matrix():
        omega = np.random.normal(0, 1 / np.sqrt(2), 8)
        # omega = np.random.uniform(0, 1 / np.sqrt(2), size=8)
        hermitian_matrix = np.zeros((3, 3), dtype=np.complex128)

        for i in range(1, 9):
            hermitian_matrix += omega[i - 1] * Tgen(i)
        return la.expm(1j * hermitian_matrix)

    # Populate U
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        U[x, y, z, t, mu] = random_su3_matrix()
    return U


@njit(complex128[:, :](complex128, complex128[:, :]), fastmath=True)
def expMatrix(idtau, P):
    method = 2

    if method == 1:
        M = idtau * P
        evals, evecs = np.linalg.eig(M)
        eD = np.diag(np.exp(evals))
        eM = evecs @ eD @ np.linalg.inv(evecs)
    if method == 2:
        # 3x3 identity matrix
        Id = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=complex128)
        # exponential expansion
        eM = (
            Id
            + idtau * P
            + idtau**2 * P @ P / 2
            + idtau**3 * P @ P @ P / 6
            + idtau**4 * P @ P @ P @ P / 24
            + idtau**5 * P @ P @ P @ P @ P / 120
        )

    return eM


@njit(
    complex128[:, :](
        int64, int64, int64, int64, int64, complex128[:, :, :, :, :, :, :]
    ),
    fastmath=True,
)
def staple(x, y, z, t, mu, U):

    """Calculate the contribution (interaction) of the three links sorrounding the link that we want to update"""
    # njit() works well!

    staple_start = np.array(
        (
            (0 + 0j, 0 + 0j, 0 + 0j),
            (0 + 0j, 0 + 0j, 0 + 0j),
            (0 + 0j, 0 + 0j, 0 + 0j),
        )
    )
    # staple_start = np.empty((3, 3)) + 0j

    a_mu = [0, 0, 0, 0]
    a_mu[mu] = 1

    for nu in range(4):

        if nu != mu:
            # nu = 0
            # while nu < mu:
            a_nu = [0, 0, 0, 0]
            a_nu[nu] = 1

            # product of 6 matrices
            staple_start += (
                U[
                    (x + a_mu[0]) % Ns,
                    (y + a_mu[1]) % Ns,
                    (z + a_mu[2]) % Ns,
                    (t + a_mu[3]) % Nt,
                    nu,
                ]
                @ U[
                    (x + a_nu[0]) % Ns,
                    (y + a_nu[1]) % Ns,
                    (z + a_nu[2]) % Ns,
                    (t + a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[x, y, z, t, nu].conj().T
            )

            staple_start += (
                U[
                    (x + a_mu[0] - a_nu[0]) % Ns,
                    (y + a_mu[1] - a_nu[1]) % Ns,
                    (z + a_mu[2] - a_nu[2]) % Ns,
                    (t + a_mu[3] - a_nu[3]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_nu[0]) % Ns,
                    (y - a_nu[1]) % Ns,
                    (z - a_nu[2]) % Ns,
                    (t - a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_nu[0]) % Ns,
                    (y - a_nu[1]) % Ns,
                    (z - a_nu[2]) % Ns,
                    (t - a_nu[3]) % Nt,
                    nu,
                ]
            )
            # nu += 1

    return staple_start


@njit(float64(int64, int64, complex128[:, :, :, :, :, :, :], float64), fastmath=True)
def Sgauge(R, T, U, beta):

    """Wilson action"""

    somma = 0
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for nu in range(4):

                        a_nu = [0, 0, 0, 0]
                        a_nu[nu] = 1

                        # while nu < mu:
                        for mu in range(4):

                            if nu != mu:

                                a_mu = [0, 0, 0, 0]
                                a_mu[mu] = 1
                                i = 0
                                j = 0

                                loop = np.identity(su3) + 0j

                                a_nu = [0, 0, 0, 0]
                                a_nu[nu] = 1

                                for i in range(R):

                                    loop = np.dot(
                                        loop,
                                        U[
                                            (x + i * a_mu[0]) % Ns,
                                            (y + i * a_mu[1]) % Ns,
                                            (z + i * a_mu[2]) % Ns,
                                            (t + i * a_mu[3]) % Nt,
                                            mu,
                                        ],
                                    )

                                for j in range(T):
                                    loop = np.dot(
                                        loop,
                                        U[
                                            (x + T * a_mu[0] + j * a_nu[0]) % Ns,
                                            (y + T * a_mu[1] + j * a_nu[1]) % Ns,
                                            (z + T * a_mu[2] + j * a_nu[2]) % Ns,
                                            (t + T * a_mu[3] + j * a_nu[3]) % Nt,
                                            nu,
                                        ],
                                    )

                                # sono questi due for loops che non mi convincono affatto!
                                # almeno non per loop di Wilson più grandi della singola plaquette

                                for i in range(R - 1, -1, -1):
                                    loop = np.dot(
                                        loop,
                                        U[
                                            (x + i * a_mu[0] + R * a_nu[0]) % Ns,
                                            (y + i * a_mu[1] + R * a_nu[1]) % Ns,
                                            (z + i * a_mu[2] + R * a_nu[2]) % Ns,
                                            (t + i * a_mu[3] + R * a_nu[3]) % Nt,
                                            mu,
                                        ]
                                        .conj()
                                        .T,
                                    )

                                for j in range(T - 1, -1, -1):
                                    loop = np.dot(
                                        loop,
                                        U[
                                            (z + j * a_nu[0]) % Ns,
                                            (y + j * a_nu[1]) % Ns,
                                            (z + j * a_nu[2]) % Ns,
                                            (t + j * a_nu[3]) % Nt,
                                            nu,
                                        ]
                                        .conj()
                                        .T,
                                    )
                                Identity = np.array(
                                    (
                                        (1 + 0j, 0 + 0j, 0 + 0j),
                                        (0 + 0j, 1 + 0j, 0 + 0j),
                                        (0 + 0j, 0 + 0j, 1 + 0j),
                                    )
                                )
                                somma += (
                                    np.trace(Identity - loop)
                                ).real  # gattringer eq. (3.4)

    return beta * somma / 6


@njit(complex128[:, :](int64), fastmath=True)
def Tgen(a):
    """Generator of su(3) algebra"""
    # njt() works well!
    if a == 1:
        T = np.array((([0, 1, 0]), ([1, 0, 0]), ([0, 0, 0])), dtype=complex128)
    elif a == 2:
        T = np.array((([0, -1j, 0]), ([1j, 0, 0]), ([0, 0, 0])), dtype=complex128)
    elif a == 3:
        T = np.array((([1, 0, 0]), ([0, -1, 0]), ([0, 0, 0])), dtype=complex128)
    elif a == 4:
        T = np.array((([0, 0, 1]), ([0, 0, 0]), ([1, 0, 0])), dtype=complex128)
    elif a == 5:
        T = np.array((([0, 0, -1j]), ([0, 0, 0]), ([1j, 0, 0])), dtype=complex128)
    elif a == 6:
        T = np.array((([0, 0, 0]), ([0, 0, 1]), ([0, 1, 0])), dtype=complex128)
    elif a == 7:
        T = np.array((([0, 0, 0]), ([0, 0, -1j]), ([0, 1j, 0])), dtype=complex128)
    elif a == 8:
        T = np.array(
            (([1, 0, 0]), ([0, 1, 0]), ([0, 0, -2])), dtype=complex128
        ) / np.sqrt(3)
    return T / 2


@njit()
def ta(mat):
    """Traceless antihermitian part of a matrix"""
    rows, cols = mat.shape

    aux = np.copy(mat)
    trace = 0 + 0j
    one_by_three = 1 / 3

    for i in range(3):
        for j in range(3):
            mat[i][j] = 0.5 * (aux[i][j] - np.conj(aux[j][i]))

    trace = np.trace(mat)
    trace *= one_by_three

    for i in range(3):
        mat[i][i] -= trace


@njit(complex128[:, :, :, :, :, :, :](), fastmath=True)
def initialMomenta():
    """Initializes momenta with a gaussian distribution
    ref. [4]"""
    # njit() works well! velocissimo
    factor_c = 1  # (Ns**3 * Nt) #decreasing this => decreases the energy fluctiations from configurations
    P = np.zeros((Ns, Ns, Ns, Nt, 4, 3, 3), dtype=np.complex128)

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        rand = np.random.normal(loc=0, scale=1 / np.sqrt(2), size=8)
                        for a in range(1, 9):
                            P[x, y, z, t, mu] += factor_c * rand[a - 1] * Tgen(a)

    # it's traceless by construction

    return P


@njit(
    complex128[:, :](
        complex128[:, :, :, :, :, :, :],
        complex128[:, :],
        float64,
        int64,
        int64,
        int64,
        int64,
        int64,
    ),
    fastmath=True,
)
def ForceTerm(U, staple, beta, x, y, z, t, mu):
    """Computes the force term of the Hamiltonian for all of 8 generators which will
    update the momenta in the leapfrog algorithm"""
    # njit() works well! velocissimo
    F = np.zeros((3, 3), dtype=np.complex128)
    F = 1j * (
        (beta / 3)
        * (U[x, y, z, t, mu] @ staple - staple.conj().T @ U[x, y, z, t, mu].conj().T)
    )
    # for a in range(1, 9):
    #     F += (
    #         (-beta / 6)
    #         * Tgen(a)
    #         * np.trace(
    #             1j
    #             * Tgen(a)
    #             * (
    #                 U[x, y, z, t, mu] @ staple
    #                 - staple.conj().T @ U[x, y, z, t, mu].conj().T
    #             )
    #         )
    #     )  # gattringer eq. 8.42 'We have used that i(U A − A† U†) is traceless and hermitian and thus is a
    # linear combination of the Tj'

    ta(F)  # F is traceless and antihermitian

    return F


######################################################LeapFrog Evolution######################################################
@njit(
    (
        complex128[:, :, :, :, :, :, :],
        complex128[:, :, :, :, :, :, :],
        float64,
        float64,
    ),
    fastmath=True,
)
def move_P(U, P, beta, deltat):

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):

                        F = ForceTerm(
                            U,
                            staple(x, y, z, t, mu, U),
                            beta,
                            x,
                            y,
                            z,
                            t,
                            mu,
                        )

                        P[x, y, z, t, mu] += (deltat) * F

    # F is traceless, element of the Lie algebra su(3)


@njit(
    (
        complex128[:, :, :, :, :, :, :],
        complex128[:, :, :, :, :, :, :],
        float64,
        float64,
    ),
    fastmath=True,
)
def move_U(U, P, beta, deltat):
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        Iu = expMatrix(deltat, 1j * P[x, y, z, t, mu])
                        U[x, y, z, t, mu] = Iu @ U[x, y, z, t, mu]

    # U is an element of the Lie group SU(3)


#######################################################################################################################à
@njit(
    complex128(
        complex128[:, :, :, :, :, :, :], complex128[:, :, :, :, :, :, :], float64
    )
)
def Hamiltonian(U, P, beta):
    """Computes the Hamiltonian of the system"""
    # njit() works well!
    H = 0
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        H += 0.5 * np.trace(P[x, y, z, t, mu] @ P[x, y, z, t, mu])

    H += Sgauge(1, 1, U, beta)

    return H


@njit(float64(complex128[:, :, :, :, :, :, :], float64), fastmath=True)
def HybridMonteCarlo(U, beta):
    """Hybrid Monte Carlo algorithm"""
    print("Executing beta = ", beta)

    smean = []

    for _ in range(N_conf):
        P = initialMomenta()
        print("configuration: ", _)

        Uold = U.copy()
        Pold = P.copy()
        Unew = U.copy()
        Pnew = P.copy()

        Hold = Hamiltonian(Uold, Pold, beta)

        # initial step

        move_U(Unew, Pnew, beta, dtau / 2)

        for _ in range(1, Nstep - 1):
            move_P(Unew, Pnew, beta, dtau)
            move_U(Unew, Pnew, beta, dtau)

            # print("hamiltoniana allo step", _, Hamiltonian(Unew, Pnew, beta))

        # final step
        move_P(Unew, Pnew, beta, dtau)
        move_U(Unew, Pnew, beta, dtau / 2)
        Pnew = -Pnew

        Hnew = Hamiltonian(Unew, Pnew, beta)
        # non lo soooo perché viene così grande deltaH?

        deltaH = Hnew.real - Hold.real

        print("deltaH:", deltaH)

        ## global metropolis step
        if deltaH < 0:
            U = Unew
            P = Pnew
            print("accepted")
        else:
            if np.random.uniform(0, 1) < (np.exp(-deltaH)):
                U = Unew
                P = Pnew
                print("accepted")
            else:
                U = Uold
                P = Pold

        ## P is traceless

        Wilson = WilsonAction(R, T, U)
        smean.append(Wilson)
        print("WilsonAction:", Wilson)

    smean = np.array(smean)
    return np.mean(smean)


if __name__ == "__main__":

    multiproc = 0
    # U = another_initializeLattice()
    U = initialize_lattice(1)

    # for _ in range(10):
    #     print("thermalize with HB...", _)
    #     U = HB_updating_links(6.7, U)

    if multiproc == 1:
        multiproc = True
    else:
        multiproc = False

    if multiproc:

        with multiprocessing.Pool(processes=len(beta_vec)) as pool:
            partial_HybridMonteCarlo = partial(HybridMonteCarlo, U)
            results = pool.map(partial_HybridMonteCarlo, beta_vec)
            pool.close()
            pool.join()

        plt.figure()
        plt.scatter(beta_vec, results, marker="x", color="red")
        plt.show()
        np.savetxt("results.txt", results)

    HybridMonteCarlo(U, 1.7)
