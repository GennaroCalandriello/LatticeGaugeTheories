import numpy as np
from numba import njit
import matplotlib.pyplot as plt

from functions import *

su2 = 2
N = 6

"""This code refers to the thesis entitled: Lattice Simulation of SU(2) Multi Higgs fields"""


sx = np.array(((0, 1), (1, 0)), complex)
sy = np.array(((0, -1j), (1j, 0)), complex)
sz = np.array(((1, 0), (0, -1)), complex)
s0 = np.identity(su2)


# Higgs parameters to controll and calibrate

Lambda = 0.1
k = 0.1
epsilon = 0.2
xi = 1 - epsilon


# @njit()
def SU2SingleMatrix():

    r0 = np.random.uniform(-0.5, 0.5)
    x0 = np.sign(r0) * np.sqrt(1 - epsilon ** 2)

    r = np.random.random((3)) - 0.5
    x = epsilon * r / np.linalg.norm(r)

    SU2Matrix = x0 * np.identity(2) + 1j * x[0] * sx + 1j * x[1] * sy + 1j * x[2] * sz

    return SU2Matrix


def initialize_fields(start):

    """Initialize gauge configuration and Higgs field"""

    U = np.zeros((N, N, N, N, 4, su2, su2), complex)
    UHiggs = U.copy()

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        UHiggs[t, x, y, z, mu] = SU2SingleMatrix()

                        if start == 0:
                            U[t, x, y, z, mu] = np.identity(su2)
                        if start == 1:
                            U[t, x, y, z, mu] = SU2SingleMatrix()

    return U, UHiggs


# @njit()
def Higgs_staple_calculus(t, x, y, z, mu, U, phi):

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

    HiggsCoupled = staple_start + 2 * k * (
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


# @njit()
def Updating_Higgs(U, phi, beta):

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):
                        phi[t, x, y, z, mu] = HB_Higgs(t, x, y, z, mu, U, phi)

    return phi


# @njit()
def Updating_gauge_configuration(U, phi, beta):

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        HiggsStaple = Higgs_staple_calculus(t, x, y, z, mu, U, phi)

                        U[t, x, y, z, mu] = HB_gauge(HiggsStaple, beta)

    return U


# @njit()
def HB_Higgs(t, x, y, z, mu, U, phi):

    """Non so se questo algoritmo è corretto"""

    V = generateV(t, x, y, z, mu, U, phi)

    z = np.array((0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j))
    phi = z.copy()

    for i in range(len(z)):

        z[i] = rand(np.random.uniform(0, 1), np.random.uniform(0, 1))

    for j in range(len(z)):

        phi[j] = z[j] - V[j] / xi

    phiupdated = np.array(
        (
            (phi[0] + 1j * phi[3], phi[2] + 1j * phi[1]),
            (-phi[2] + 1j * phi[1], phi[0] - 1j * phi[3]),
        )
    )
    print(np.linalg.det(phiupdated))

    return phiupdated


# @njit()
def rand(x1, x2):

    return np.sqrt(-np.log(x1) / xi) * np.cos(np.pi * 2 * x2)


# @njit()
def generateV(t, x, y, z, mu, U, phi):

    """equation 3.36"""
    Vtemp = np.array(((0 + 0j, 0 + 0j), (0 + 0j, 0 + 0j)))

    for nu in range(4):

        a_nu = [0, 0, 0, 0]
        a_nu[nu] = 1

        Vtemp += (
            phi[
                (t + a_nu[0]) % N,
                (x + a_nu[1]) % N,
                (y + a_nu[2]) % N,
                (z + a_nu[3]) % N,
                nu,
            ]
            .conj()
            .T
            @ U[t, x, y, z, nu].conj().T
            + phi[
                (t - a_nu[0]) % N,
                (x - a_nu[1]) % N,
                (y - a_nu[2]) % N,
                (z - a_nu[3]) % N,
                nu,
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

    V0 = (k / 2) * np.trace(s0 @ Vtemp)
    V1 = (k / 2) * np.trace(sx @ Vtemp)
    V2 = (k / 2) * np.trace(sy @ Vtemp)
    V3 = (k / 2) * np.trace(sz @ Vtemp)

    return np.array((V0, V1, V2, V3))


# @njit()
def HB_gauge(staple, beta):

    w = normalize(getA(staple))
    a = np.sqrt(np.abs(np.linalg.det(staple)))
    wbar = quaternion((w))

    if a != 0:
        xw = quaternion(sampleA(beta, a))

        xx = xw @ wbar.conj().T  ###!!!!warning!!!

        return xx

    else:
        return SU2SingleMatrix()


# @njit()
def normalize(v):

    return v / np.sqrt(v.dot(v))


# @njit()
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

    return somma / (6 * N ** 4)


def prova(phi):

    print(np.trace(phi[0, 0, 0, 0, 0].conj().T @ phi[0, 0, 0, 0, 0]))


# @njit()
def completeHiggsAction(R, T, U, phi):

    Wilson = WilsonAction(R, T, U)
    somma = 0
    phi1, phi2, phi3 = 0, 0, 0

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    # phi1 = np.identity(su2) + 0j
                    # phi2 = phi1.copy()
                    # phi3 = phi1.copy()

                    for mu in range(4):

                        a_mu = [0, 0, 0, 0]
                        a_mu[mu] = 1

                        phi1 += (
                            Lambda
                            * (
                                (1 / 2)
                                * np.trace(
                                    phi[t, x, y, z, mu].conj().T @ phi[t, x, y, z, mu]
                                )
                                - 1
                            )
                            ** 2
                        )

                        phi2 += 0.5 * np.trace(
                            phi[t, x, y, z, mu].conj().T @ phi[t, x, y, z, mu]
                        )

                        for nu in range(4):

                            a_nu = [0, 0, 0, 0]
                            a_nu[nu] = 1

                            phi3 += -k * np.trace(
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
                        # print("phi1=", phi1, "phi2=", phi2, "phi3=", phi3)

    return phi1 + phi2 + phi3 + Wilson


if __name__ == "__main__":

    U, phi = initialize_fields(1)

    measures = 10
    R, T = 1, 1

    betavec = np.linspace(0.1, 8.0, 10).tolist()

    results = []

    # print(np.trace(phi[0, 0, 0, 0, 0].conj().T @ U[0, 0, 0, 0, 0] @ phi[0, 0, 0, 0, 0]))

    for b in betavec:
        print("exe for beta: ", b)

        obs = []

        for i in range(measures):

            U = Updating_gauge_configuration(U, phi, b)
            phi = Updating_Higgs(U, phi, b)

            temp = completeHiggsAction(R, T, U, phi)
            # print(temp)
            # obs.append(temp)

        results.append(np.mean(obs))

