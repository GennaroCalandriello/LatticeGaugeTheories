import multiprocessing
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit, float64, int64

from randomSU2 import *
from algebra import *
from functions import *
import parameters as par

import warnings
from numba.core.errors import NumbaPerformanceWarning

# Disable the performance warning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

su3 = par.su3
su2 = par.su2
Ns = par.Ns
Nt = par.Nt
epsilon = par.epsilon
# N = par.N
pool_size = par.pool_size
N_conf = par.N_conf
measures = N_conf
beta_vec = par.beta_vec

# What kind of update do u want, bitch?
overrelax = False
metropolis = False
heatbath = True


# Wilson Loop extension
R = par.R
T = par.T

sx = np.array(((0, 1), (1, 0)), dtype=np.complex128)
sy = np.array(((0, -1j), (1j, 0)), dtype=np.complex128)
sz = np.array(((1, 0), (0, -1)), dtype=np.complex128)

#
@njit()
def HB_updating_links(beta, U):

    """Update each link via the heatbath algorithm"""

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for mu in range(4):

                        # l'idea è creare 3 heatbath 'annidati'
                        # in modo da estrarre R dall'heatbath di W=UA,
                        # S dall'heathbath di W=RUA e T dall'heathbath di W=SRUA

                        Utemp = U[x, y, z, t, mu]

                        # staple must be initialized for each calculus

                        A = staple_calculus(x, y, z, t, mu, U)
                        W = np.dot(Utemp, A)

                        a = det_calculus(W)

                        if a != 0:
                            # else:
                            r = heatbath_SU3(W, beta, "R")
                            R = np.array(
                                (
                                    (r[0, 0], r[0, 1], 0),
                                    (r[1, 0], r[1, 1], 0),
                                    (0, 0, 1),
                                )
                            )

                            W = np.dot(R, W)
                            s = heatbath_SU3(W, beta, "S")

                            S = np.array(
                                (
                                    (s[0, 0], 0, s[0, 1]),
                                    (0, 1, 0),
                                    (s[1, 0], 0, s[1, 1]),
                                )
                            )

                            W = np.dot(S, W)

                            tt = heatbath_SU3(W, beta, "T")

                            T = np.array(
                                (
                                    (1, 0, 0),
                                    (0, tt[0, 0], tt[0, 1]),
                                    (0, tt[1, 0], tt[1, 1]),
                                )
                            )
                            # U'=TSRU
                            Uprime = T @ S @ R @ Utemp
                            U[x, y, z, t, mu] = Uprime

                        else:
                            U[x, y, z, t, mu] = SU3SingleMatrix()

    return U


@njit()
def OverRelaxation_(U):

    """Ettore Vicari, An overrelaxed Monte Carlo algorithm
    for SU(3) lattice gauge theories"""

    # Funziona benissimo bitches!!!

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for mu in range(4):

                        Utemp = U[x, y, z, t, mu]

                        A = staple_calculus(x, y, z, t, mu, U)
                        a = np.sqrt(np.linalg.det(A))
                        A = A / a
                        Adagger = A.conj().T

                        Atemp = Adagger @ A
                        H = matrixsqrt(Atemp)  # H is Hermitean matrix, controlled

                        O = A @ np.linalg.inv(H)
                        det_O = np.linalg.det(O)

                        if round(det_O.real) == 1:

                            I_alpha = (np.identity(su3) + 0j) * (1)
                            O = np.dot(O, I_alpha)

                        if round(det_O.real) == -1:

                            I_alpha = (np.identity(su3) + 0j) * (-1)
                            O = np.dot(O, I_alpha)

                        V = diagonalization(H)

                        Urefl = V @ Utemp @ O @ V.conj().T
                        Urefl = reflectionSU3(Urefl, np.random.randint(0, 4))
                        Uprime = V.conj().T @ Urefl @ V @ O.conj().T
                        U[x, y, z, t, mu] = Uprime
    return U


@njit()
def matrixsqrt(M):

    evals, evecs = np.linalg.eig(M)
    sqrt = evecs @ np.diag(np.sqrt(evals)) @ np.linalg.inv(evecs)

    return sqrt


@njit()
def invMatrix(M):
    evals, evecs = np.linalg.eig(M)
    inv = evecs @ np.diag(1 / evals) @ np.linalg.inv(evecs)
    return inv


@njit()
def heatbath_SU3(W, beta, subgrp, kind=1):

    """Execute Heat Bath on each of three submatrices of SU(3) (R, S; and T) through the quaternion representation. HB
    extracts the SU(2) matrix directly from the distribution that leaves the Haar measure invariant"""

    if subgrp == "R":
        Wsub = np.array(((W[0, 0], W[0, 1]), (W[1, 0], W[1, 1])))
    if subgrp == "S":
        Wsub = np.array(((W[0, 0], W[0, 2]), (W[2, 0], W[2, 2])))
    if subgrp == "T":
        Wsub = np.array(((W[1, 1], W[1, 2]), (W[2, 1], W[2, 2])))

    if kind == 1:
        w = getA(Wsub)
        a = np.sqrt(np.abs(np.linalg.det(Wsub)))
        wbar = quaternion(normalize(w))

        if a != 0:
            xw = quaternion(sampleA(beta * 2 / 3, a))

            xx = xw @ wbar.conj().T  ###!!!!warning!!!

            return xx

        else:
            return SU2SingleMatrix()

    if kind == 2:
        return sample_HB_SU2(Wsub, beta)


@njit()
def GS(A):

    """Gram Schmidt orthogolanization. It's not necessary"""

    if len(A[0]) == 3:

        u = np.array((A[0, 0], A[0, 1], A[0, 2]))
        v = np.array((A[1, 0], A[1, 1], A[1, 2]))

        u = normalize(u)
        v -= u * u.dot(v)
        v = normalize(v)
        v -= u * u.dot(v)
        v = normalize(v)
        uxv = np.cross(u, v)
        uxv = normalize(uxv)

        A[0] = u
        A[1] = v
        A[2] = uxv

    if len(A[0]) == 2:
        u = np.array((A[0, 0], A[0, 1]))
        v = np.array((A[1, 0], A[1, 1]))

        u = normalize(u)
        v -= u * u.dot(v)
        v = normalize(v)
        v -= u * u.dot(v)
        v = normalize(v)

        A[0] = u
        A[1] = v

    return A


@njit()
def WilsonAction(R, T, U):

    """Name says"""

    somma = 0
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for nu in range(4):

                        a_nu = [0, 0, 0, 0]
                        a_nu[nu] = 1

                        # while nu < mu:
                        for mu in range(nu + 1, 4):
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

                            somma += np.trace(loop).real / su3

    return somma / (6 * Ns**3 * Nt)


@njit()
def calculate_S(link, stapla, beta):

    ans = np.trace(np.dot(stapla, link)).real
    return -beta * ans / su3


@njit()
def Metropolis(U, beta, hits):

    """Execution of Metropolis, checking every single site 10 times."""

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):

                        staple = staple_calculus(x, y, z, t, mu, U)

                        for _ in range(hits):

                            old_link = U[x, y, z, t, mu].copy()
                            S_old = calculate_S(old_link, staple, beta)

                            su3matrix = SU3SingleMatrix()
                            new_link = np.dot(su3matrix, old_link)
                            S_new = calculate_S(new_link, staple, beta)
                            dS = S_new - S_old

                            if dS < 0:
                                U[x, y, z, t, mu] = new_link
                            else:
                                if np.exp(-dS) > np.random.uniform(0, 1):
                                    U[x, y, z, t, mu] = new_link
                                else:
                                    U[x, y, z, t, mu] = old_link
    return U


@njit()
def diagonalization(Matrix):

    _, eVec = np.linalg.eig(Matrix)
    V = eVec
    return V


def main(beth):

    U = initialize_lattice(1)
    print("Shape of my heart is a song by Sting", U.shape)
    print("exe for beta = ", beth)
    obs = []
    obs2 = []
    obs3 = []

    for _ in range(measures):

        if heatbath:
            U = HB_updating_links(beth, U)

        if metropolis:
            U = Metropolis(U, beth, hits=10)

        if overrelax:
            for _ in range(1):
                U = OverRelaxation_(U)

        # two different wilson loops
        temp = WilsonAction(R, T, U)

        temp2 = WilsonAction(3, 3, U)
        temp3 = WilsonAction(2, 2, U)

        print("wilson_11 action", temp)

        obs.append(temp)
        obs2.append(temp2)
        obs3.append(temp3)

    W11 = np.mean(obs)
    W22 = np.mean(obs3)
    W33 = np.mean(obs2)

    return W11, W22, W33


def checkPath(pathlista):

    for p in pathlista:
        p = str(p)
        print("path ", p)
        if os.path.exists(p) == False:
            os.makedirs(p)
        else:
            shutil.rmtree(p)
            os.makedirs(p)


if __name__ == "__main__":

    import time

    execution = True
    path = "data/Wilson_loop/"  # change this path to your own

    if execution:
        print("Update with:")
        if heatbath:
            print("heabath")
        if overrelax:
            print("OR")
        if metropolis:
            print("Metro")

        idecorrel = par.idecorrel
        # which kind of link update would you like to use?

        checkPath(path)

        Smean = []
        Smean2 = []
        Smean3 = []

        staple_start = np.zeros((su3, su3), dtype=np.complex128)
        quat = np.zeros((su2, su2), dtype=np.complex128)

        start = time.time()

        with multiprocessing.Pool(processes=len(beta_vec)) as pool:
            results = pool.map(main, beta_vec)
            for result in results:
                Smean.append(result[0])
                Smean2.append(result[1])
                Smean3.append(result[2])

            pool.close()
            pool.join()

        np.savetxt(f"{path}/mean_action_L_{Ns}_W11.txt", Smean)
        np.savetxt(f"{path}/mean_action_L_{Ns}_W22.txt", Smean3)
        np.savetxt(f"{path}/mean_action_L_{Ns}_W33.txt", Smean2)

        print(f"Execution time: {round(time.time() - start, 2)} s")
        W11 = np.loadtxt(f"{path}/mean_action_L_{Ns}_W11.txt")
        W22 = np.loadtxt(f"{path}/mean_action_L_{Ns}_W22.txt")
        W33 = np.loadtxt(f"{path}/mean_action_L_{Ns}_W33.txt")

        plt.figure()
        plt.title(" SU(3) Wilson loop for N = 6", fontsize=18)
        plt.scatter(beta_vec, Smean, s=3, c="r", label=r"$W_{11}$")
        plt.scatter(beta_vec, Smean2, s=3, c="b", label=r"$W_{22}$")
        plt.scatter(beta_vec, Smean3, s=3, c="g", label=r"$W_{33}$")
        plt.xlabel(r"$\beta$", fontsize=16)
        plt.ylabel(r"$\langle W_{R,T} \rangle$ ", fontsize=16)
        plt.legend()
        plt.show()
