import numpy as np
import multiprocessing
from functools import partial

# from hybridMonteCarlo import *
from parameters import *
from HB_OR_SU3 import *
from functions import *
from numba import njit, float64, int64, complex128, boolean


"""Hybrid Monte Carlo and Wilson flow
ref.:
[1] Bonati, D'Elia: Comparison of the gradient flow with cooling in SU(3) pure gauge theory

"""


@njit(boolean(complex128[:, :]), fastmath=True)
def antiHermitianVerify(A):
    if A.any() == (-A.conj().T).any():
        return True
    else:
        return False


@njit(float64(int64, int64, complex128[:, :, :, :, :, :, :], float64), fastmath=True)
def Sgauge(R, T, U, beta):

    """Wilson action
    Bonati, D'Elia: Comparison of the gradient flow with cooling in SU(3) pure gauge theory"""

    somma = 0
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        A = staple(x, y, z, t, mu, U)
                        somma += -(beta / su3) * np.trace(
                            (U[x, y, z, t, mu] @ A.conj().T).real
                        )
    return somma


@njit(
    complex128[:, :](
        complex128[:, :, :, :, :, :, :],
        complex128[:, :, :, :, :, :, :],
        complex128[:, :],
        float64,
        int64,
        int64,
        int64,
        int64,
        int64,
    )
)
def Force(U, P, A, beta, x, y, z, t, mu):
    """similar to Vdot, from ref. [1] eq. (7). It seems work!"""

    traceless = True

    Fmu = -1j * (
        (beta / 12) * (U[x, y, z, t, mu] @ A - A.conj().T @ U[x, y, z, t, mu].conj().T)
        - (1 / (2 * su3))
        * np.trace(U[x, y, z, t, mu] @ A - A.conj().T @ U[x, y, z, t, mu].conj().T)
    )

    if traceless:

        for i in range(3):
            Fmu[i, i] -= (1 / 3) * np.trace(Fmu)

        return Fmu

    else:
        return Fmu


@njit(
    complex128[:, :](
        complex128[:, :, :, :, :, :, :],
        complex128[:, :, :, :, :, :, :],
        complex128[:, :],
        float64,
        int64,
        int64,
        int64,
        int64,
        int64,
    )
)
def Vdot(U, V, A, beta, x, y, z, t, mu):
    """[1] eqs. (3) & (7)"""

    # dmuS = np.zeros((3, 3), dtype=np.complex128)
    # Vdotmu = np.zeros((3, 3), dtype=np.complex128)

    Omega_mu = U[x, y, z, t, mu] @ A.conj().T
    Vdotmu = (
        -(beta / 12)
        * (
            0.5 * (Omega_mu - Omega_mu.conj().T)
            - (1 / (2 * su3)) * np.trace(Omega_mu - Omega_mu.conj().T)
        )
        @ V[x, y, z, t, mu]
    )

    # for mu in range(4):
    #     Vdotmu += -dmuS @ V[x, y, z, t, mu]

    # ta(Vdotmu)
    return Vdotmu


def gradientFlow():
    pass


@njit(
    (
        complex128[:, :, :, :, :, :, :],
        complex128[:, :, :, :, :, :, :],
        float64,
        float64,
    ),
    fastmath=True,
)
def move_P_due(U, P, beta, dtau):
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        A = staple(x, y, z, t, mu, U)
                        Vdotmu = Force(U, P, A, beta, x, y, z, t, mu)
                        P[x, y, z, t, mu] -= dtau * Vdotmu


"""this is a test script"""


def move_U(U, P, beta, deltat):
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        Iu = expMatrix(deltat, 1j * P[x, y, z, t, mu])
                        U[x, y, z, t, mu] = Iu @ U[x, y, z, t, mu]

    # U is an element of the Lie group SU(3)


@njit(
    complex128(
        complex128[:, :, :, :, :, :, :], complex128[:, :, :, :, :, :, :], float64
    )
)
def Hamiltonian(U, P, beta):

    H = 0

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        H += np.trace(P[x, y, z, t, mu] @ P[x, y, z, t, mu].conj().T)
    H += Sgauge(R, T, U, beta)

    return H


@njit(boolean(complex128, complex128))
def MetropolisTest(Hold, Hnew):
    """Metropolis acceptance test"""

    deltaH = (Hnew - Hold).real
    deltaH = deltaH / (Ns**3 * Nt * 2)
    if deltaH < 0:
        print("accett")
        return True

    elif np.random.uniform(0, 1) < np.exp(-deltaH):
        print("accett")
        return True

    else:
        return False


@njit(
    (complex128[:, :, :, :, :, :, :], complex128[:, :, :, :, :, :, :], float64),
    fastmath=True,
)
def leapFrogIntegrator(U, P, beta):
    """this is the leapfrog integrator
    see Gattringer, Lang: Quantum Chromodynamics on the Lattice"""

    # first step
    move_P_due(U, P, beta, dtau / 2)

    # leapfrog
    for _ in range(Nstep):
        if _ % 10 == 0:
            print("time step = ", _)
        move_U(U, P, beta, dtau)
        move_P_due(U, P, beta, dtau)

    # last steps
    move_U(U, P, beta, dtau)
    move_P_due(U, P, beta, dtau / 2)


@njit(float64[:](complex128[:, :, :, :, :, :, :], float64), fastmath=True)
def HMC_definitivo_siSpera(U, beta):
    """this is the definitivo version of HMC, IFFFFfF it works!"""
    Wilsons = np.zeros((N_conf), dtype=np.float64)
    print("Execution for beta = ", beta)
    for conf in range(N_conf):
        print("configurationssssssssssss = ", conf)

        Uold = U.copy()
        Unew = U.copy()
        P = initialMomenta()

        # integration of flow equations through leaprfrog
        leapFrogIntegrator(Unew, P, beta)

        # metropolis a/r-----------------------------------------------
        Hold = Hamiltonian(Uold, P, beta)
        Hnew = Hamiltonian(Unew, P, beta)
        print("deltaH = ", Hnew - Hold)

        if conf == 0:  # accept the first configuration a priori
            U = Unew.copy()
            Wilson = WilsonAction(R, T, U)
            print("Wilson action = ", Wilson)
            Wilsons[conf] = Wilson

        else:  # metropolis test
            Metrotest = MetropolisTest(Hold, Hnew)

            if Metrotest:
                U = Unew.copy()
            elif not Metrotest:
                U = Uold.copy()
            # U = Unew.copy()
            # Wilson = WilsonAction(R, T, U)
            print("Wilson action = ", Wilson)
            Wilsons[conf] = Wilson
        # --------------------------------------------------------------------------------------------
    return Wilsons


if __name__ == "__main__":
    U = initialize_lattice(1)
    print("initialwislon action", WilsonAction(R, T, U))
    # P = initialMomenta()

    test = 2

    if test == 1:
        Wilsons = []
        Uhb = U.copy()
        Uhmc = U.copy()

        for conf in range(N_conf):
            P = initialMomenta()
            Uhb = HB_updating_links(beta, U)

            # wilson action for hb
            WilsonHb = WilsonAction(R, T, Uhb)
            # Hold = Hamiltonian(Uhmc, P)
            # one leapfrog step
            move_P_due(Uhmc, P, beta, dtau / 2)
            for _ in range(20):
                print("time step = ", _)
                move_U(Uhmc, P, beta, dtau)
                move_P_due(Uhmc, P, beta, dtau)

            move_U(Uhmc, P, beta, dtau)
            move_P_due(Uhmc, P, beta, dtau / 2)

            # Wilspn action for hmc
            Wilsonhmc = WilsonAction(R, T, Uhmc)

            print("WilsonHb = ", WilsonHb)
            print("Wilsonhmc = ", Wilsonhmc)

    elif test == 2:

        with multiprocessing.Pool(processes=len(beta_vec)) as pool:
            temp = partial(HMC_definitivo_siSpera, U)
            Ws = np.array(pool.map(temp, beta_vec))
        print("Wilson action")

        S_mean = []
        for wbeta in Ws:
            S_mean.append(np.mean(wbeta))

        plt.figure()
        plt.scatter(beta_vec, S_mean)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$S_W$")
        plt.show()
        np.savetxt("data/S_mean_HMC.txt", S_mean)
        # HMC_definitivo_siSpera(U)
