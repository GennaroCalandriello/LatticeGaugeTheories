import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit

from randomSU2 import *
from algebra import *

su3 = 3
su2 = 2
epsilon = 0.24
N = 6
pool_size = N ** 4

sx = np.array(((0, 1), (1, 0)), complex)
sy = np.array(((0, -1j), (1j, 0)), complex)
sz = np.array(((1, 0), (0, -1)), complex)

"""Questo cazzo di programma con njit funziona, mo se bell 
e buon non funziona più non ha senso, rinuncia agli studi e
 iscriviti a scienze della formalina"""


def SU3_pool_generator(pool_size):

    # following pag 83 Gattringer
    su2_pool = SU2_pool_generator(pool_size * 3, epsilon=epsilon)
    su3_pool = np.zeros((pool_size, su3, su3), complex)

    for i in range(int(pool_size / 2)):  # half pool RST and half RST. conj.T
        r = su2_pool[i]
        s = su2_pool[i + int(pool_size / 2)]
        t = su2_pool[i + 2 * int(pool_size / 2)]

        R = np.array(((r[0, 0], r[0, 1], 0), (r[1, 0], r[1, 1], 0), (0, 0, 1)))
        S = np.array(((s[0, 0], 0, s[0, 1]), (0, 1, 0), (s[1, 0], 0, s[1, 1])))
        T = np.array(((1, 0, 0), (0, t[0, 0], t[0, 1]), (0, t[1, 0], t[1, 1])))

        su3_pool[i] = R @ S @ T
        su3_pool[i + int(pool_size / 2)] = (su3_pool[i].conj().T).copy()

    return su3_pool


def initialize_lattice(start, N):

    U = np.zeros((N, N, N, N, 4, su3, su3), complex)
    su3_pool = SU3_pool_generator(pool_size=pool_size)
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):
                        if start == 0:
                            U[t, x, y, z, mu] = np.identity(su3)
                        if start == 1:
                            U[t, x, y, z, mu] = su3_pool[
                                np.random.randint(0, pool_size)
                            ]
    return U


@njit()
def det_calculus(W):

    # if len(W[:, 0]) == 3:
    #     det = (
    #         W[0, 0] * W[1, 1] * W[2, 2]
    #         + W[0, 1] * W[1, 2] * W[2, 0]
    #         + W[0, 2] * W[1, 0] * W[2, 1]
    #         - W[0, 2] * W[1, 1] * W[2, 0]
    #         - W[0, 1] * W[1, 0] * W[2, 2]
    #         - W[0, 0] * W[1, 2] * W[2, 1]
    #     )

    # if len(W[:, 0]) == 2:
    #     det = W[0, 0] * W[1, 1] - W[0, 1] * W[1, 0]

    det1 = np.linalg.det(W)
    # if det == det1:
    #     print("uguali")

    return det1


@njit()
def staple_calculus(t, x, y, z, mu, U, staple_start):
    # staple_start = np.zeros((su3, su3), complex)
    a_mu = [0, 0, 0, 0]
    a_mu[mu] = 1

    for nu in range(4):

        if mu == nu:
            continue
        else:
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

    return staple_start


@njit()
def initialize_staple(staple_start):

    for i in range(su3):
        for j in range(su3):
            staple_start[i, j] = 0 + 0j

    return staple_start


pool_size = N ** 5
su3_pool = SU3_pool_generator(pool_size)
su2_pool = SU2_pool_generator(pool_size, epsilon)
#
@njit()
def updating_links(beta, U, N, staple_start):

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        # l'idea è creare 3 heatbath annidati in modo da estrarre R dall'heatbath di W=UA, S dall'heathbath di W=RUA e T dall'heathbath di W=SRUA
                        # staple_start = initialize_staple(staple_start)
                        Utemp = U[t, x, y, z, mu]

                        staple_start = initialize_staple(staple_start)
                        A = staple_calculus(t, x, y, z, mu, U, staple_start)
                        W = Utemp @ A
                        a = det_calculus(W)
                        if a.real == 0:
                            # print(a)
                            continue

                        else:
                            r = heatbath_SU3(W, beta)
                            R = np.array(
                                (
                                    (r[0, 0], r[0, 1], 0),
                                    (r[1, 0], r[1, 1], 0),
                                    (0, 0, 1),
                                )
                            )

                            W = R @ W
                            s = heatbath_SU3(W, beta)
                            S = np.array(
                                (
                                    (s[0, 0], 0, s[0, 1]),
                                    (0, 1, 0),
                                    (s[1, 0], 0, s[1, 1]),
                                )
                            )
                            W = S @ W

                            tt = heatbath_SU3(W, beta)
                            T = np.array(
                                (
                                    (1, 0, 0),
                                    (0, tt[0, 0], tt[0, 1]),
                                    (0, tt[1, 0], tt[1, 1]),
                                )
                            )
                            # U'=TSRU

                            Uprime = T @ S @ R @ Utemp
                            Uprime = gram_schmidt(Uprime)

                            U[t, x, y, z, mu] = Uprime

    return U


@njit()
def heatbath_SU3(W, beta):

    # print(W)
    w = getA(W)

    a = np.sqrt(np.abs(det_calculus(W)))
    wbar = quaternion(w)

    if a != 0:
        xw = quaternion(sampleA(beta, a))
        xx = xw * wbar.conj().T

    return xx


@njit()
def quaternion(vec):

    """produces quaternion from a vector of complex and real numbers"""

    # vec = vec / np.linalg.norm(vec)

    a11 = vec[0] + vec[3] * 1j
    a12 = vec[2] + vec[1] * 1j
    a21 = -vec[2] + vec[1] * 1j
    a22 = vec[0] - vec[3] * 1j

    # quat[0, 0] = a11
    # quat[1, 0] = a21
    # quat[0, 1] = a12
    # quat[1, 1] = a22
    # print(quat)
    quat = np.array(((a11, a12), (a21, a22)))
    # # print("arrrrrrrrrrr", quat1)
    # if quat.any() == quat1.any():
    #     print("yessir")
    # print(quat, "quetzalcoatl")

    return quat


@njit()
def sampleA(a, beta):

    # choose a0 with P(a0) ~ sqrt(1 - a0^2) * exp(beta * k * a0)
    w = np.exp(-2 * beta * a)
    xtrial = np.random.uniform(0, 1) * (1 - w) + w
    a0 = 1 + np.log(xtrial) / (beta * a)

    while (1 - a0 ** 2) < np.random.uniform(0, 1):
        xtrial = np.random.uniform(0, 1) * (1 - w) + w
        a0 = 1 + np.log(xtrial) / (beta * a)

    r = np.sqrt(1 - a0 ** 2)
    a1 = np.random.normal()
    a2 = np.random.normal()
    a3 = np.random.normal()

    while (a1 ** 2 + a2 ** 2 + a3 ** 2) > 0.7:
        a1 = np.random.normal()
        a2 = np.random.normal()
        a3 = np.random.normal()

    norm = np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)

    a1 = a1 * r / norm
    a2 = a2 * r / norm
    a3 = a3 * r / norm

    avec = np.array((a0, a1, a2, a3))
    # print("avemaria", avec)

    return avec


@njit()
def getA(W):

    a0 = ((W[0, 0] + W[1, 1])).real / 2
    a1 = ((W[0, 1] + W[1, 0])).imag / 2
    a2 = ((W[0, 1] - W[1, 0])).real / 2
    a3 = ((W[0, 0] - W[1, 1])).imag / 2
    Avector = [a0, a1, a2, a3]

    return Avector


@njit()
def gram_schmidt(A):

    (n, m) = A.shape

    for i in range(m):

        q = A[:, i]  # i-th column of A

        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]

        # if np.array_equal(q, np.zeros(q.shape)):
        #     raise np.linalg.LinAlgError(
        #         "The column vectors are not linearly independent"
        #     )

        # normalize q
        q = q / np.sqrt(np.dot(q, q))

        # write the vector back in the matrix
        A[:, i] = q

    return A


@njit()
def W_11(S, U):
    somma = 0
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):
                        a_mu = [0, 0, 0, 0]

                        nu = 0
                        while nu < mu:
                            a_nu = [0, 0, 0, 0]

                            loop = np.identity(su3) + 0j

                            for i in range(4):
                                if i == mu:
                                    a_mu[i] = 1
                                else:
                                    a_mu[i] = 0

                                if i == nu:
                                    a_nu[i] = 1
                                else:
                                    a_nu[i] = 0

                            s = 0
                            while s < S:
                                loop = (
                                    loop
                                    @ U[
                                        (t + a_mu[s]) % N,
                                        (x + a_mu[s]) % N,
                                        (y + a_mu[s]) % N,
                                        (z + a_mu[s]) % N,
                                        mu,
                                    ]
                                )
                                s += 1

                            s = 0
                            while s < S:
                                loop = (
                                    loop
                                    @ U[
                                        (t + a_mu[S] + a_nu[s]) % N,
                                        (x + a_mu[S] + a_nu[s]) % N,
                                        (y + a_mu[S] + a_nu[s]) % N,
                                        (z + a_mu[S] + a_nu[s]) % N,
                                        nu,
                                    ]
                                )
                                s += 1
                            s = 0
                            while s < S:
                                loop = (
                                    loop
                                    @ U[
                                        (t + a_mu[s] + a_nu[S]) % N,
                                        (x + a_mu[s] + a_nu[S]) % N,
                                        (y + a_mu[s] + a_nu[S]) % N,
                                        (z + a_mu[s] + a_nu[S]) % N,
                                        mu,
                                    ]
                                    .conj()
                                    .T
                                )
                                s -= 1
                            s = 0
                            while s < S:
                                loop = (
                                    loop
                                    @ U[
                                        (t + a_nu[s]) % N,
                                        (x + a_nu[s]) % N,
                                        (y + a_nu[s]) % N,
                                        (z + a_nu[s]) % N,
                                        nu,
                                    ]
                                    .conj()
                                    .T
                                )
                                s -= 1
                            s = 0

                            somma += np.trace(loop).real / su3
                            nu += 1

    return somma / (6 * N ** 4)


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

                        mu = 0
                        while mu < nu:

                            a_mu = [0, 0, 0, 0]
                            a_mu[mu] = 1

                            ii = 0
                            jj = 0

                            temp = np.identity(su3) + 0j

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

                            somma += (np.trace(temp)).real / su3
                            mu += 1

    return somma / (6 * N ** 4)


if __name__ == "__main__":
    measures = 20
    idecorrel = 1
    beta_vec = np.linspace(1.0, 10.1, 8)
    U = initialize_lattice(1, N)

    Smean = []

    staple_start = np.zeros((su3, su3), complex)
    quat = np.zeros((su2, su2), complex)

    for beth in beta_vec:
        print("exe for beta = ", beth)
        obs = []
        for _ in range(measures):
            for _ in range(idecorrel):
                U = updating_links(beth, U, N, staple_start)
            # temp = S(1, 1, U)
            temp = W_11(1, U)
            print(temp)
            obs.append(temp)
        Smean.append(np.mean(obs))

    plt.scatter(beta_vec, Smean)
    plt.show()

