import numpy as np
import matplotlib.pyplot as plt
from numba import njit, float64, jit

from algebra import *

from statistics import *
import time

su2 = 2
N = 10

SU2_pool_size = N ** 4
epsilon = 0.24


def SU2_pool_generator(SU2_pool_size, epsilon):

    SU2_pool = np.zeros((SU2_pool_size, su2, su2), np.complex)
    lista = []

    for i in range(SU2_pool_size):

        r0 = np.random.uniform(-0.5, 0.5)
        x0 = np.sign(r0) * np.sqrt(1 - epsilon ** 2)
        r = np.random.uniform(0, 1, size=3)
        x = epsilon * r / np.linalg.norm(r)

        SU2_pool[i] = (
            np.identity(su2) * x0 + 1j * x[0] * sx + 1j * x[1] * sy + 1j * x[2] * sz
        )

    return SU2_pool


def init_lattice(start, N):
    random_pool = SU2_pool_generator(SU2_pool_size=SU2_pool_size, epsilon=epsilon)

    U = np.zeros((N, N, N, N, 4, su2, su2), np.complex)
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):
                        if np.random.uniform(0, 1) > start:  # cold start
                            U[t, x, y, z, mu] = np.identity(su2)
                        else:
                            U[t, x, y, z, mu] = random_pool[  # hot start
                                int(np.random.randint(0, SU2_pool_size - 1))
                            ]
    return U


@njit()
def staple_calculus(t, x, y, z, mu, beta, U, staple_start):
    # staple_start = np.zeros((su2, su2), np.complex)
    a_mu = [0, 0, 0, 0]
    a_mu[mu] = 1

    # for nu in range(4):

    #     if nu != mu:
    nu = 0
    while nu < mu:
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
        nu += 1

    return staple_start


@njit()
def lambda2generator(beta, a):

    r_g = 1 - np.random.uniform(0, 1, size=3)
    lambda2 = (np.log(r_g[0]) + (np.cos(2 * np.pi * r_g[1]) ** 2) * np.log(r_g[2])) / (
        -2 * a * beta
    )

    return lambda2


@njit()
def initialize_staple(staple_start):

    for j in range(su2):
        for k in range(su2):
            staple_start[j, k] = 0 + 0j

    return staple_start


@njit()
def generate_SU2_matrix(X, sx, sy, sz):
    r0 = np.random.uniform(-0.5, 0.5)
    x0 = np.sign(r0) * np.sqrt(1 - epsilon ** 2)
    r = np.random.uniform(0, 1, size=3)
    xran = epsilon * r / np.linalg.norm(r)

    X = (
        np.identity(su2) * x0
        + 1j * xran[0] * sx
        + 1j * xran[1] * sy
        + 1j * xran[2] * sz
    )
    return X


@njit()
def heatbath_update(beta, U, staple_start, sx, sy, sz, N):

    # sx = np.array(((0, 1), (1, 0)))
    # sy = np.array(((0, -1j), (1j, 0)))
    # sz = np.array(((1, 0), (0, -1)))

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        staple_start = initialize_staple(staple_start)
                        A = staple_calculus(t, x, y, z, mu, beta, U, staple_start)
                        a = np.sqrt(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])

                        # a = np.sqrt(np.linalg.det(A))

                        if a == 0:

                            # random_pool = SU2_pool_generator(
                            #     SU2_pool_size=4, epsilon=epsilon
                            # )
                            # U[t, x, y, z, mu] = random_pool[np.random.randint(0, 4)]
                            r0 = np.random.uniform(-0.5, 0.5)
                            x0 = np.sign(r0) * np.sqrt(1 - epsilon ** 2)
                            r = np.random.uniform(0, 1, size=3)
                            xran = epsilon * r / np.linalg.norm(r)

                            U[t, x, y, z, mu] = (
                                np.identity(su2) * x0
                                + 1j * xran[0] * sx
                                + 1j * xran[1] * sy
                                + 1j * xran[2] * sz
                            )

                        else:
                            lambda2 = lambda2generator(beta, a)

                            while ((1 - np.random.uniform(0, 1))) ** 2 > (
                                1 - lambda2.real
                            ):
                                lambda2 = lambda2generator(beta, a)

                            xvec = np.random.uniform(-1, 1, size=3)

                            while (xvec[0] ** 2 + xvec[1] ** 2 + xvec[2] ** 2) > 1:
                                xvec = np.random.uniform(-1, 1, size=3)

                            # building X for SU(2) equazione 4.24 Gattringer
                            x0 = 1 - 2 * lambda2

                            xv = xvec  # lambda2 ** 0.5 * xvec / (np.linalg.norm(xvec))

                            X = (
                                x0 * np.identity(su2)
                                + 1j * xv[0] * sx
                                + 1j * xv[1] * sy
                                + 1j * xv[2] * sz
                            )

                            U[t, x, y, z, mu] = np.dot(X, (A / a).conj().T)

    return U


@njit()
def delta_S(U_old, U_new, staple, beta):

    dS = np.trace(np.dot((U_new - U_old), staple)).real
    dS = -beta * dS / su2

    return dS


@njit()
def Metropolis_update(staple_start, staple2, beta, U, sx, sy, sz):
    idecorrel = 20
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        staple_start = initialize_staple(staple_start)
                        A = staple_calculus(t, x, y, z, mu, beta, U, staple_start)

                        for idec in range(idecorrel):

                            U_old = U[t, x, y, z, mu].copy()
                            X = initialize_staple(staple2)
                            X = generate_SU2_matrix(X, sx, sy, sz)
                            U_new = np.dot(X, U_old)
                            dS = delta_S(U_old, U_new, A, beta)
                            aa = beta / N
                            if dS < 0.0 or np.exp(-dS / (2 * aa)) > np.random.rand():
                                U[t, x, y, z, mu] = U_new

                            # else:
                            #     U[t, x, y, z, mu] = U_old

    return U


@njit()
def S(I, J, U):  # costruisce l'azione di Wilson

    # !!!!!!!!!!!  WARNING  !!!!!!!!!!
    somma = 0
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    # mmmmmh da rivedere################
                    # U[N - 1, x, y, z] = U[0, x, y, z]
                    # U[t, N - 1, y, z] = U[t, 0, y, z]
                    # U[t, x, N - 1, z] = U[t, x, 0, z]
                    # U[t, x, y, N - 1] = U[t, x, y, 0]
                    # ###################################

                    for nu in range(4):

                        a_nu = [0, 0, 0, 0]
                        a_nu[nu] = 1

                        for mu in range(nu + 1, 4):

                            a_mu = [0, 0, 0, 0]
                            a_mu[mu] = 1

                            ii = 0
                            jj = 0

                            temp = np.identity(su2) + 1j

                            for ii in range(0, I):
                                temp = np.dot(
                                    temp,
                                    U[
                                        (t + ii * a_mu[0]) % N,
                                        (x + ii * a_mu[1]) % N,
                                        (y + ii * a_mu[2]) % N,
                                        (z + ii * a_mu[3]) % N,
                                        mu,
                                    ],
                                )

                            for jj in range(0, J):
                                temp = np.dot(
                                    temp,
                                    U[
                                        (t + (ii + 1) * a_mu[0] + jj * a_nu[0]) % N,
                                        (x + (ii + 1) * a_mu[1] + jj * a_nu[1]) % N,
                                        (y + (ii + 1) * a_mu[2] + ii * a_nu[2]) % N,
                                        (z + (ii + 1) * a_mu[3] + jj * a_nu[3]) % N,
                                        nu,
                                    ],
                                )

                            for ii in range(ii, -1, -1):
                                temp = np.dot(
                                    temp,
                                    U[
                                        (t + ii * a_mu[0] + (jj + 1) * a_nu[0]) % N,
                                        (x + ii * a_mu[1] + (jj + 1) * a_nu[1]) % N,
                                        (y + ii * a_mu[2] + (jj + 1) * a_nu[2]) % N,
                                        (z + ii * a_mu[3] + (jj + 1) * a_nu[3]) % N,
                                        mu,
                                    ]
                                    .conj()
                                    .T,
                                )

                            for jj in range(jj, -1, -1):
                                temp = np.dot(
                                    temp,
                                    U[
                                        (t + jj * a_nu[0]) % N,
                                        (x + jj * a_nu[1]) % N,
                                        (y + jj * a_nu[2]) % N,
                                        (z + jj * a_nu[3]) % N,
                                        nu,
                                    ]
                                    .conj()
                                    .T,
                                )

                            somma += (np.trace(temp)).real / su2

    return somma / (6 * N ** 4)


if __name__ == "__main__":
    sx = np.array(((0, 1), (1, 0)), complex)
    sy = np.array(((0, -1j), (1j, 0)), complex)
    sz = np.array(((1, 0), (0, -1)), complex)
    exe = True

    beta_array = np.linspace(0.1, 12, 4)

    idecorrel = 1
    measures = 3

    # print(U)

    obsbeta = []
    # U = heatbath_update(beta_array[2], U, staple_start, sx, sy, sz, N)
    # action=S(1, 1, U)
    # print(action)
    if exe:
        for beth in beta_array:
            U = init_lattice(1, N)
            obs = []
            print(f"Execution for beta = {beth}")

            for m in range(measures):

                staple_start = np.zeros((su2, su2), complex)
                staple2 = staple_start.copy()
                U = Metropolis_update(staple_start, staple2, beth, U, sx, sy, sz)
                # U = heatbath_update(beth, U, staple_start, sx, sy, sz, N)

                summ = S(1, 1, U)
                print(f"Heat-bath measure {m}")
                # prodo, summ = WilsonLoop(U, beta)
                obs.append(summ)
                print(f"Measure of action? {summ}")
                # print(U)

            obsbeta.append(np.mean(obs))

        plt.figure()
        plt.plot(beta_array, obsbeta, "ro")
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"$<S_P>/(6 N^4)$")
        plt.show()

