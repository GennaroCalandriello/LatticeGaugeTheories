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

references:
[1] S. Schaefer, R. Sommer, and F. Virotta, “Critical slowing down and error analysis in lattice QCD simulations,” Nucl. Phys. B, vol. 845, no. 1, pp. 93–114, 2011.   
[2] M. Lüscher, “Topology, the Wilson flow and the HMC algorithm,” PoS, vol. LATTICE2013, p. 016, 2014.
[3] M. Lüscher, “Properties and uses of the Wilson flow in lattice QCD,” JHEP, vol. 08, p. 071, 2010.
[4] R. Gupta et al "Comparison of Update Algorithms for Pure Gauge SU(3)", http://cds.cern.ch/record/186962/files/CM-P00052019.pdf
[5] Kennedy, Pendleton "ACCEPTANCES AND AUTOCORRELATIONS IN HYBRID MONTE CARLO
[6] https://arxiv.org/abs/hep-lat/0502007v3"

alcune regole:
1. nelle funzioni se c'è U viene prima di tutti gli altri argomenti
2. ordine generale (U, P, beta, x, y, z, t, mu):


NOTA IMPORTANTE: njit(argomentooutput(argomentiinput), fastmath =True) funziona velocissimo
ma le funzioni devono essere strutturate per cacare solo 1 argomento alla volta
deltaH must be of order O(dtau^2) [6]
"""


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
    return somma  # / (Ns**3 * Nt * 2)


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


@njit(
    complex128(
        complex128[:, :, :, :, :, :, :],
        complex128[:, :, :, :, :, :, :],
        float64,
        boolean,
    )
)
def Hamiltonian(U, P, beta, onlyP):
    """Hamiltonian for the HMC algorithm"""

    H = 0
    if onlyP:
        H = 0
        for x in range(Ns):
            for y in range(Ns):
                for z in range(Ns):
                    for t in range(Nt):
                        for mu in range(4):

                            H += np.trace(
                                P[x, y, z, t, mu] @ P[x, y, z, t, mu].conj().T
                            )
        return H

    else:

        for x in range(Ns):
            for y in range(Ns):
                for z in range(Ns):
                    for t in range(Nt):
                        for mu in range(4):

                            H += (
                                0.5
                                * (
                                    np.trace(
                                        P[x, y, z, t, mu] @ P[x, y, z, t, mu].conj().T
                                    )
                                )
                                / (Ns**3 * Nt)
                            )
        H += Sgauge(R, T, U, beta)

        return H / (Ns**3 * Nt * 6)


@njit(fastmath=True)
def UnitarizeConfiguration(C):

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        unitarize(C[x, y, z, t, mu])


@njit(boolean(complex128, complex128))
def MetropolisTest(Hold, Hnew):
    """Metropolis acceptance test"""

    deltaH = (Hnew - Hold).real
    deltaH = deltaH
    if deltaH < 0:
        print("accett")
        return True

    elif np.random.uniform(0, 1) < np.exp(-deltaH):
        print("accett")
        return True

    else:
        print("rifiut")
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
    for _ in range(Nstep - 1):
        if _ % 10 == 0:
            print("time step = ", _)
        move_U(U, P, beta, dtau)
        move_P_due(U, P, beta, dtau)

    # last steps
    move_U(U, P, beta, dtau)


@njit(float64[:](complex128[:, :, :, :, :, :, :], float64), fastmath=True)
def HMC_definitivo_siSpera(U, beta):
    """this is the definitivo version of HMC, IFFFFfF it works!"""
    Wilsons = np.zeros((N_conf), dtype=np.float64)
    print("Execution for beta = ", beta)
    acceptance_rate = 0

    for conf in range(N_conf):
        print("configurationssssssssssss = ", conf)

        Uold = U.copy()
        Unew = U.copy()
        P = initialMomenta()
        Pold = P.copy()
        HPold = Hamiltonian(Uold, Pold, beta, onlyP=True)

        # integration of flow equations through leaprfrog
        leapFrogIntegrator(Unew, P, beta)
        UnitarizeConfiguration(P)

        # metropolis a/r-----------------------------------------------
        Hold = Hamiltonian(Uold, P, beta, onlyP=False)
        Hnew = Hamiltonian(Unew, P, beta, onlyP=False)

        # normalize?
        Hold = Hold
        Hnew = Hnew

        # print("deltaH = ", Hnew - Hold)

        # print("Only P", HpNew) #10368

        if conf == 0:  # accept the first configuration a priori
            U = Unew.copy()
            Wilson = WilsonAction(R, T, U)
            print("Wilson action = ", Wilson)
            Wilsons[conf] = Wilson

        else:  # metropolis test
            Metrotest = MetropolisTest(Hold, Hnew)

            if Metrotest:
                U = Unew.copy()
                acceptance_rate += 1
            elif not Metrotest:
                U = Uold.copy()
            # U = Unew.copy()
            # Wilson = WilsonAction(R, T, U)
            print("Wilson action = ", Wilson)

        ####senza metropolis
        # U = Unew.copy()
        # Wilson = WilsonAction(R, T, U)

        Wilsons[conf] = Wilson
        # --------------------------------------------------------------------------------------------
    print("Acceptance rate for beta", beta, (acceptance_rate / (conf)))

    return Wilsons


@njit(float64[:](complex128[:, :, :, :, :, :, :], float64), fastmath=True)
def comparisonWithHB(U, beta):
    """For a comparison"""

    print("executione heatbattttth beta = ", beta)
    WilsonsHB = []
    for conf in range(N_conf):
        U = HB_updating_links(beta, U)
        WilsonsHB.append(WilsonAction(R, T, U))
    WilsonsHB = np.array(WilsonsHB)

    return WilsonsHB


if __name__ == "__main__":
    U = initialize_lattice(1)
    Uhb = U.copy()
    print("initialwislon action", WilsonAction(R, T, U))
    # P = initialMomenta()

    test = 2

    if test == 1:
        """compare with Heat-Bath withouth Metropolis test"""
        Wilsons = []
        Uhb = U.copy()
        Uhmc = U.copy()

        for conf in range(N_conf):
            P = initialMomenta()
            Uhb = HB_updating_links(beta, U)

            # wilson action for hb
            WilsonHb = WilsonAction(R, T, Uhb)
            # Hold = Hamiltonian(Uhmc, P)

            # one leapfrog step-------
            move_P_due(Uhmc, P, beta, dtau / 2)
            for _ in range(20):
                print("time step = ", _)
                move_U(Uhmc, P, beta, dtau)
                move_P_due(Uhmc, P, beta, dtau)

            move_U(Uhmc, P, beta, dtau)
            move_P_due(Uhmc, P, beta, dtau / 2)
            # ----------------------------

            # Wilspn action for hmc
            Wilsonhmc = WilsonAction(R, T, Uhmc)

            print("WilsonHb = ", WilsonHb)
            print("Wilsonhmc = ", Wilsonhmc)

    elif test == 2:
        """HMC with multiprocessing. Plot the mean of the Wilson action
        as a function of beta"""

        with multiprocessing.Pool(processes=len(beta_vec)) as pool:
            temp = partial(HMC_definitivo_siSpera, U)
            Ws = np.array(pool.map(temp, beta_vec))

            temp1 = partial(comparisonWithHB, Uhb)
            WsHB = np.array(pool.map(temp1, beta_vec))

            pool.close()
            pool.join()

        S_mean = []
        S_meanHB = []
        for b in range(len(Ws)):
            S_mean.append(np.mean(Ws[b]))
            S_meanHB.append(np.mean(WsHB[b]))

        plt.figure()
        plt.scatter(beta_vec, S_mean, marker="x", color="red", label="HMC")
        plt.scatter(beta_vec, S_meanHB, marker="x", color="blue", label="HB")
        plt.xlabel(r"$\beta$")
        plt.legend()
        plt.ylabel(r"$S_W$")
        plt.show()
        np.savetxt("data/S_mean_HMC.txt", S_mean)
        # HMC_definitivo_siSpera(U)
