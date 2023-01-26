import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import numpy.random as ran

from functions import *

su2 = 2
N = 6

"""This code refers to the works: 

1. "Numerical Simulations of
    Gauge-Higgs Models on the Lattice", of Jochen Heitger, 1997
2. "Lattice Simulation of SU(2) Multi Higgs fields", Mark B. Wurtz
3. "On the phase diagram of the Higgs SU(2) model", C. Bonati, M. D'Elia"""


sx = np.array(((0, 1), (1, 0)), complex)
sy = np.array(((0, -1j), (1j, 0)), complex)
sz = np.array(((1, 0), (0, -1)), complex)
s0 = np.identity(su2) + 0j


# Higgs parameters

Lambda = 0.0005
# k = 0.2
epsilon = 0.2
xi = 1.08  # 1 - epsilon


def initialize_fields(start):

    """Initialize gauge configuration and Higgs field"""

    U = np.zeros((N, N, N, N, 4, su2, su2), complex)
    UHiggs = U.copy()

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        s = SU2SingleMatrix()
                        UHiggs[t, x, y, z, mu] = s

                        if start == 0:
                            U[t, x, y, z, mu] = np.identity(su2)
                        if start == 1:
                            U[t, x, y, z, mu] = SU2SingleMatrix()

    return U, UHiggs


@njit()
def Higgs_staple_calculus(t, x, y, z, mu, U, phi, k, beta, couplingHiggs):

    """Calculate the contribution (interaction) of the three links sorrounding the link that we want to update"""
    # staple_start = np.zeros((su3, su3), complex)
    staple_start = np.array(((0 + 0j, 0 + 0j), (0 + 0j, 0 + 0j)))
    # staple_start = np.empty((3, 3)) + 0j

    a_mu = [0, 0, 0, 0]
    a_mu[mu] = 1

    for nu in range(4):

        if mu != nu:
            # nu = 0
            # while nu < mu:
            a_nu = [0, 0, 0, 0]
            a_nu[nu] = 1

            # product of 6 matrices
            staple_start += (
                U[
                    (t + a_mu[0]) % N,
                    (x + a_mu[1]) % N,
                    (y + a_mu[2]) % N,
                    (z + a_mu[3]) % N,
                    nu,
                ]
                @ U[
                    (t + a_nu[0]) % N,
                    (x + a_nu[1]) % N,
                    (y + a_nu[2]) % N,
                    (z + a_nu[3]) % N,
                    mu,
                ]
                .conj()
                .T
                @ U[t, x, y, z, nu].conj().T
            )

            staple_start += (
                U[
                    (t + a_mu[0] - a_nu[0]) % N,
                    (x + a_mu[1] - a_nu[1]) % N,
                    (y + a_mu[2] - a_nu[2]) % N,
                    (z + a_mu[3] - a_nu[3]) % N,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (t - a_nu[0]) % N,
                    (x - a_nu[1]) % N,
                    (y - a_nu[2]) % N,
                    (z - a_nu[3]) % N,
                    mu,
                ]
                .conj()
                .T
                @ U[
                    (t - a_nu[0]) % N,
                    (x - a_nu[1]) % N,
                    (y - a_nu[2]) % N,
                    (z - a_nu[3]) % N,
                    nu,
                ]
            )
            # nu += 1
        else:
            continue

    """equation 3.34"""
    if couplingHiggs:

        HiggsCoupled = 0.5 * beta * staple_start + k * (
            phi[
                (t + a_mu[0]) % N,
                (x + a_mu[1]) % N,
                (y + a_mu[2]) % N,
                (z + a_mu[3]) % N,
                mu,
            ]
            @ phi[t, x, y, z, mu].conj().T
        )

        return HiggsCoupled

    else:
        return 0.5 * beta * staple_start
    # print("staple", np.linalg.det(staple_start))
    # print("higgsstaple", HiggsCoupled - beta * staple_start)


@njit()
def SU2SingleMatrix():

    r0 = np.random.uniform(-0.5, 0.5)
    x0 = np.sign(r0) * np.sqrt(1 - epsilon ** 2)

    r = np.random.random((3)) - 0.5
    x = epsilon * r / np.linalg.norm(r)

    SU2Matrix = x0 * np.identity(2) + 1j * x[0] * sx + 1j * x[1] * sy + 1j * x[2] * sz

    return SU2Matrix


###########################################################UPDATING FUNCTIONS####################################
@njit()
def HB_Higgs(t, x, y, z, mu, U, phi, k):

    """Equation (A.14) of 1."""

    V0 = 0 + 0j
    V1 = V0
    V2 = V0
    V3 = V0

    for nu in range(4):

        a_nu = [0, 0, 0, 0]
        a_nu[nu] = 1

        V0 += np.trace(
            phi[
                (t + a_nu[0]) % N,
                (x + a_nu[1]) % N,
                (y + a_nu[2]) % N,
                (z + a_nu[3]) % N,
                nu,
            ]
            .conj()
            .T
            @ U[t, x, y, z, nu]
            @ s0
            + s0
            @ U[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
            @ phi[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
        )

        V1 += np.trace(
            phi[
                (t + a_nu[0]) % N,
                (x + a_nu[1]) % N,
                (y + a_nu[2]) % N,
                (z + a_nu[3]) % N,
                nu,
            ]
            .conj()
            .T
            @ U[t, x, y, z, nu]
            @ sx
            + sx
            @ U[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
            @ phi[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
        )

        V2 += np.trace(
            phi[
                (t + a_nu[0]) % N,
                (x + a_nu[1]) % N,
                (y + a_nu[2]) % N,
                (z + a_nu[3]) % N,
                nu,
            ]
            .conj()
            .T
            @ U[t, x, y, z, nu]
            @ sy
            + sy
            @ U[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
            @ phi[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
        )

        V3 += np.trace(
            phi[
                (t + a_nu[0]) % N,
                (x + a_nu[1]) % N,
                (y + a_nu[2]) % N,
                (z + a_nu[3]) % N,
                nu,
            ]
            .conj()
            .T
            @ U[t, x, y, z, nu]
            @ sz
            + sz
            @ U[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
            @ phi[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
        )
    phi0 = V0 * (1j * k / 2) + np.sqrt(-(1 / xi) * np.log(ran.uniform(0, 1))) * np.sin(
        2 * np.pi * ran.uniform(0, 1)
    )
    phi1 = V1 * (1j * k / 2) + np.sqrt(-(1 / xi) * np.log(ran.uniform(0, 1))) * np.sin(
        2 * np.pi * ran.uniform(0, 1)
    )
    phi2 = V2 * (1j * k / 2) + np.sqrt(-(1 / xi) * np.log(ran.uniform(0, 1))) * np.sin(
        2 * np.pi * ran.uniform(0, 1)
    )
    phi3 = V3 * (1j * k / 2) + np.sqrt(-(1 / xi) * np.log(ran.uniform(0, 1))) * np.sin(
        2 * np.pi * ran.uniform(0, 1)
    )

    phivec = np.array((phi0, phi1, phi2, phi3))
    phivec = normalize(phivec)

    phinew = np.array(
        (
            (phivec[0] + 1j * phivec[3], phivec[2] + 1j * phivec[1]),
            (-phivec[2] + 1j * phivec[1], phivec[0] - 1j * phivec[3]),
        )
    )

    return phinew


@njit()
def HB_gauge(staple, beta):

    w = normalize(getA(staple))
    a = np.sqrt(np.abs(np.linalg.det(staple)))
    wbar = quaternion((w))

    if a != 0:
        xw = quaternion(sampleA(beta, a))

        xx = xw @ wbar.conj().T

        return xx

    else:
        return SU2SingleMatrix()


@njit()
def reflectionFunction(t, x, y, z, mu, phi, U, k, comp, phicomp):

    """Reflection function from A.16 of reference 1."""

    if comp == 0:
        s = s0
    if comp == 1:
        s = sx
    if comp == 2:
        s = sy
    if comp == 3:
        s = sz

    phinew = 0

    for nu in range(4):
        a_nu = [0, 0, 0, 0]
        a_nu[nu] = 1
        phinew += np.trace(
            phi[
                (t + a_nu[0]) % N,
                (x + a_nu[1]) % N,
                (y + a_nu[2]) % N,
                (z + a_nu[3]) % N,
                nu,
            ]
            .conj()
            .T
            @ U[t, x, y, z, nu]
            @ s
            + s
            @ U[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
            @ phi[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
            ]
        )
    if comp == 0:
        phinew = phinew * (2 / xi) * k / 2 - phicomp
    else:
        phinew = phinew * (2 / xi) * 1j * k / 2 - phicomp

    return phinew


@njit()
def OverRelaxationHiggs(t, x, y, z, mu, phi, U, k):

    """equation A.16 reference 1."""

    phitemp = phi[t, x, y, z, mu]

    phi0 = phitemp[0, 0].real
    phi3 = phitemp[0, 0].imag
    phi2 = phitemp[1, 0].real
    phi1 = phitemp[1, 0].imag

    phi0 = reflectionFunction(t, x, y, z, mu, phi, U, k, 0, phi0)
    phi1 = reflectionFunction(t, x, y, z, mu, phi, U, k, 1, phi1)
    phi2 = reflectionFunction(t, x, y, z, mu, phi, U, k, 2, phi2)
    phi3 = reflectionFunction(t, x, y, z, mu, phi, U, k, 3, phi3)

    phivec = np.array((phi0, phi1, phi2, phi3))
    phivec = normalize(phivec)

    phinew = np.array(
        (
            (phivec[0] + 1j * phivec[3], phivec[2] + 1j * phivec[1],),
            (-phivec[2] + 1j * phivec[1], phivec[0] - 1j * phivec[3],),
        )
    )
    phi[t, x, y, z, mu] = phinew

    return phi[t, x, y, z, mu]


@njit()
def OverRelaxationSU2(t, x, y, z, mu, U, phi, beta, k):

    Utemp = U[t, x, y, z, mu]
    V = Higgs_staple_calculus(t, x, y, z, mu, U, phi, k, beta, couplingHiggs=True)
    v = np.linalg.det(V)
    # print(v)

    if v != 0:

        V = V / (np.sqrt(v))
        Uprime = V.conj().T @ Utemp.conj().T @ V.conj().T

        U[t, x, y, z, mu] = Uprime

    return U[t, x, y, z, mu]


@njit()
def Updating_Higgs(U, phi, beta, k):

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):
                        s = HB_Higgs(t, x, y, z, mu, U, phi, k)
                        # print(np.linalg.det(s))
                        phi[t, x, y, z, mu] = s

    return phi


@njit()
def OverRelaxation_update(U, phi, beta, k):

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        phi[t, x, y, z, mu] = OverRelaxationHiggs(
                            t, x, y, z, mu, phi, U, k
                        )

                        U[t, x, y, z, mu] = OverRelaxationSU2(
                            t, x, y, z, mu, U, phi, beta, k
                        )

    return U, phi


@njit()
def HeatBath_update(U, phi, beta, k):

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        HiggsStaple = Higgs_staple_calculus(
                            t, x, y, z, mu, U, phi, k, beta, couplingHiggs=True
                        )
                        Unew = HB_gauge(HiggsStaple, beta)
                        U[t, x, y, z, mu] = Unew
                        phi[t, x, y, z, mu] = HB_Higgs(t, x, y, z, mu, U, phi, k)

    return U, phi


####################################################END UPDATING FUNCTIONS##########################################


@njit()
def normalize(v):

    return v / np.sqrt(v.dot(v))


@njit()
def WilsonAction(R, T, U):

    """Name says"""

    somma = 0
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    for nu in range(4):

                        a_nu = [0, 0, 0, 0]
                        a_nu[nu] = 1

                        # while nu < mu:
                        for mu in range(nu + 1, 4):
                            a_mu = [0, 0, 0, 0]
                            a_mu[mu] = 1
                            i = 0
                            j = 0

                            loop = np.identity(su2) + 0j

                            a_nu = [0, 0, 0, 0]
                            a_nu[nu] = 1

                            for i in range(R):

                                loop = np.dot(
                                    loop,
                                    U[
                                        (t + i * a_mu[0]) % N,
                                        (x + i * a_mu[1]) % N,
                                        (y + i * a_mu[2]) % N,
                                        (z + i * a_mu[3]) % N,
                                        mu,
                                    ],
                                )

                            for j in range(T):
                                loop = np.dot(
                                    loop,
                                    U[
                                        (t + T * a_mu[0] + j * a_nu[0]) % N,
                                        (x + T * a_mu[1] + j * a_nu[1]) % N,
                                        (y + T * a_mu[2] + j * a_nu[2]) % N,
                                        (z + T * a_mu[3] + j * a_nu[3]) % N,
                                        nu,
                                    ],
                                )

                            for i in range(R - 1, -1, -1):
                                loop = np.dot(
                                    loop,
                                    U[
                                        (t + i * a_mu[0] + R * a_nu[0]) % N,
                                        (x + i * a_mu[1] + R * a_nu[1]) % N,
                                        (y + i * a_mu[2] + R * a_nu[2]) % N,
                                        (z + i * a_mu[3] + R * a_nu[3]) % N,
                                        mu,
                                    ]
                                    .conj()
                                    .T,
                                )

                            for j in range(T - 1, -1, -1):
                                loop = np.dot(
                                    loop,
                                    U[
                                        (t + j * a_nu[0]) % N,
                                        (x + j * a_nu[1]) % N,
                                        (y + j * a_nu[2]) % N,
                                        (z + j * a_nu[3]) % N,
                                        nu,
                                    ]
                                    .conj()
                                    .T,
                                )

                            somma += np.trace(loop).real / su2
    somma = somma / (6 * N ** 4)
    # print("somma", somma)

    return somma


def prova(phi):

    print(np.trace(phi[0, 0, 0, 0, 0].conj().T @ phi[0, 0, 0, 0, 0]))


@njit()
def completeHiggsAction(R, T, U, phi, k):

    """Reference 3."""

    Wilson = WilsonAction(R, T, U)

    kind = 2  # minimal action coupling with higgs

    phi1 = 0
    phi2 = 0
    phi3 = 0

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(1, 4):

                        nu = mu
                        a_nu = [0, 0, 0, 0]
                        a_nu[nu] = 1
                        phi3 += (
                            -k
                            * 0.5
                            * np.trace(
                                phi[t, x, y, z, nu].conj().T
                                @ U[
                                    (t + a_nu[0]) % N,
                                    (x + a_nu[1]) % N,
                                    (y + a_nu[2]) % N,
                                    (z + a_nu[3]) % N,
                                    nu,
                                ]
                                @ phi[
                                    (t + a_nu[0]) % N,
                                    (x + a_nu[1]) % N,
                                    (y + a_nu[2]) % N,
                                    (z + a_nu[3]) % N,
                                    nu,
                                ]
                            )
                        )

    # return (phi1 + phi2 + phi3) / (6 * N ** 4) + beta * (1 - Wilson)
    return phi3 / (4 * N ** 4) + Wilson


@njit()
def extendedHiggsAction(R, T, U, phi, k):

    Wilson = WilsonAction(R, T, U)

    kind = 2  # minimal action coupling with higgs

    phi1 = 0
    phi2 = 0
    phi3 = 0

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(1, 4):

                        phi1 += 0.5 * np.trace(
                            phi[t, x, y, z, mu].conj().T @ phi[t, x, y, z, mu]
                        )
                        phi2 += (
                            Lambda
                            * (
                                0.5
                                * np.trace(
                                    phi[t, x, y, z, mu].conj().T @ phi[t, x, y, z, mu]
                                )
                                - 1
                            )
                            ** 2
                        )

                        for nu in range(4):
                            a_nu = [0, 0, 0, 0]
                            a_nu[nu] = 1
                            phi3 += (
                                -k
                                * 2
                                * np.trace(
                                    phi[t, x, y, z, nu].conj().T
                                    @ U[t, x, y, z, nu]
                                    @ phi[
                                        (t + a_nu[0]) % N,
                                        (x + a_nu[1]) % N,
                                        (y + a_nu[2]) % N,
                                        (z + a_nu[3]) % N,
                                        nu,
                                    ]
                                )
                            )
    return (phi2) / (N ** 4) + phi1 / (4 * N ** 4) + phi3 / N ** 4 + Wilson


def main(par, measures, beta, kvec, k, R, T):

    """par specify which parameter do you want to vary during sim., k (per beta che varia), beta (per k che varia), R e T estensione del Wilson loop"""

    U, phi = initialize_fields(1)

    betavec = np.linspace(0.1, 8.0, 25).tolist()

    results = []
    susceptibility = []

    overrelax = True
    heatbath = True

    if par == "k":

        k = 0.0

        for k in kvec:

            print(f"exe for k = {k}, remaining {len(kvec)-kvec.index(k)}")

            obs = []

            for i in range(measures):

                U, phi = HeatBath_update(U, phi, beta, k)

                if overrelax:
                    for _ in range(2):
                        # phi = OverRelaxationHiggs(phi, U, k)
                        U, phi = OverRelaxation_update(U, phi, beta, k)

                temp = completeHiggsAction(R, T, U, phi, k)
                # temp = extendedHiggsAction(R, T, U, phi, k)
                # temp = WilsonAction(R, T, U)
                print(round(temp.real, 16))
                print("imaginary part:", round(temp.imag, 4))
                obs.append(temp)

            obs = np.array(obs)
            susc = N ** 4 * (np.mean(obs ** 2) - (np.mean(obs)) ** 2)
            susceptibility.append(susc)
            results.append(np.mean(obs).real)
            print("mean", np.mean(obs))

        plt.figure()
        plt.title(f"SU(2)-Higgs coupling, Heat Bath, beta = {beta}", fontsize=23)
        plt.xlabel(r"k", fontsize=16)
        # plt.ylabel(r"$\phi^2+\phi^4$", fontsize=20)
        plt.ylabel(r"$<S>$", fontsize=20)
        plt.legend([f"beta = {beta}"])
        plt.plot(kvec, results, "bo-")
        plt.show()
        # np.savetxt(f"resuts_beta_{beta}.txt", results)

        # if par == "beta":

        # beta = 0

        # for beta in betavec:

        #     print(
        #         f"exe for beta = {beta}, remaining {len(betavec)-betavec.index(beta)}"
        #     )

        #     obs = []

        #     for i in range(measures):

        #         U = Updating_gauge_configuration(U, phi, beta, k)
        #         phi = Updating_Higgs(U, phi, beta, k)

        #         # temp = completeHiggsAction(R, T, U, phi, k)
        #         temp = extendedHiggsAction(R, T, U, phi, k)
        #         print(temp)
        #         obs.append(temp)

        #     results.append(np.mean(obs).real)
        #     print("mean", np.mean(obs))

        # plt.figure()
        # plt.title(r"$\phi^2+\phi^4$  Heat Bath", fontsize=23)
        # plt.xlabel(r"$\beta$", fontsize=16)
        # plt.ylabel(r"<$S_{SU2}$+$S_{Higgs}$>", fontsize=20)
        # plt.legend([r"$\beta$ = {0.8}"])
        # plt.plot(betavec, susceptibility, "bo--")
        # plt.show()


if __name__ == "__main__":

    import time

    graph = False
    start = time.time()
    measures = 3
    beta = 2.5
    numberofk = 30
    kmin = 0.0
    kmax = 1.4
    kvec = np.linspace(kmin, kmax, numberofk).tolist()

    if graph:

        results = np.loadtxt("resuts_beta_2.5.txt")
        plt.figure()
        plt.title(f"SU(2)-Higgs coupling, Heat Bath, beta = {beta}", fontsize=23)
        plt.xlabel(r"k", fontsize=16)
        # plt.ylabel(r"$\phi^2+\phi^4$", fontsize=20)
        plt.ylabel(r"$<S>$", fontsize=20)
        plt.legend([f"beta = {beta}"])
        plt.plot(kvec, results, "bo")
        plt.show()

    else:
        main("k", measures, beta, kvec, 0, 1, 1)

