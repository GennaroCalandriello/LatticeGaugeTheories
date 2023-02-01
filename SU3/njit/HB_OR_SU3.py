import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit, float64, int64

from randomSU2 import *
from algebra import *
from functions import *
import parameters as par

su3 = par.su3
su2 = par.su2
epsilon = par.epsilon
N = par.N
pool_size = par.pool_size

sx = np.array(((0, 1), (1, 0)), complex)
sy = np.array(((0, -1j), (1j, 0)), complex)
sz = np.array(((1, 0), (0, -1)), complex)

#
@njit()
def HB_updating_links(beta, U, N):

    """Update each link via the heatbath algorithm"""

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for mu in range(4):

                        # l'idea è creare 3 heatbath 'annidati'
                        # in modo da estrarre R dall'heatbath di W=UA,
                        # S dall'heathbath di W=RUA e T dall'heathbath di W=SRUA

                        Utemp = U[t, x, y, z, mu]

                        # staple must be initialized for each calculus

                        A = staple_calculus(t, x, y, z, mu, U)
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
                            U[t, x, y, z, mu] = Uprime

                        else:
                            U[t, x, y, z, mu] = SU3SingleMatrix()

    return U


@njit()
def OverRelaxation_update(U, N):

    """Ettore Vicari, An overrelaxed Monte Carlo algorithm
        for SU(3) lattice gauge theories"""

    # I don't know if it works good!

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for mu in range(4):

                        Utemp = U[t, x, y, z, mu]

                        A = staple_calculus(t, x, y, z, mu, U)
                        a = np.sqrt((np.linalg.det(A)))

                        if a != 0:

                            # A = A / a
                            Adagger = A.conj().T
                            H = np.sqrt(Adagger @ A)
                            O = A @ (1 / H)
                            O = GramSchmidt(O, True)

                            det_O = np.linalg.det(O.conj().T).real

                            if (det_O) != 0:

                                V = diagonalization(H)

                                Uprime = V @ Utemp @ O @ V.conj().T

                                reflection = np.random.randint(1, 4)

                                Uprime = reflectionSU3(Uprime, reflection)

                                Ufinal = V.conj().T @ Uprime @ V @ O.conj().T

                                # Ufinal = GramSchmidt(Ufinal, exe=True)
                                detU = np.linalg.det(Ufinal).real

                                if round(detU) == 1:
                                    I_alpha = (np.identity(su3) + 0j) * (1)
                                    Ufinal = np.dot(Ufinal, I_alpha)

                                if round(detU) == -1:
                                    I_alpha = (np.identity(su3) + 0j) * (-1)
                                    Ufinal = np.dot(Ufinal, I_alpha)

                                U[t, x, y, z, mu] = Ufinal
                        else:
                            U[t, x, y, z, mu] = SU3SingleMatrix()

    return U


@njit()
def OverRelaxation_(U, N):

    """Ettore Vicari, An overrelaxed Monte Carlo algorithm
        for SU(3) lattice gauge theories"""

    # I don't know if it works good!

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for mu in range(4):

                        Utemp = U[t, x, y, z, mu]

                        A = staple_calculus(t, x, y, z, mu, U)
                        a = np.sqrt(np.linalg.det(A))
                        A = A / a
                        # print("dettofatto", np.linalg.det(A))
                        Adagger = A.conj().T

                        Atemp = Adagger @ A
                        H = matrixsqrt(Atemp)  # H is Hermitean matrix, controlled

                        O = A @ (1 / H)  # @ invMatrix(H)
                        # O = np.divide(A, H)
                        det_O = np.linalg.det(O)

                        if round(det_O.real) == 1:

                            I_alpha = (np.identity(su3) + 0j) * (1)
                            O = np.dot(O, I_alpha)

                        if round(det_O.real) == -1:

                            I_alpha = (np.identity(su3) + 0j) * (-1)
                            O = np.dot(O, I_alpha)

                        V = diagonalization(H)
                        # Uprime = V @ Utemp @ Otilde @ V.conj().T
                        # Uprime = reflectionSU3(Uprime, np.random.randint(0, 4))

                        Urefl = V @ Utemp @ O @ V.conj().T
                        Urefl = reflectionSU3(Urefl, np.random.randint(0, 4))
                        Uprime = V.conj().T @ Urefl @ V @ O.conj().T

                        # print(np.linalg.det(Uprime))
                        U[t, x, y, z, mu] = Uprime
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
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

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

                            # sono questi due for loops che non mi convincono affatto!
                            # almeno non per loop di Wilson più grandi della singola plaquette

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

                            somma += np.trace(loop).real / su3

    return somma / (6 * N ** 4)


@njit()
def calculate_S(link, stapla, beta):

    ans = np.trace(np.dot(stapla, link)).real
    return -beta * ans / su3


@njit()
def Metropolis(U, beta, hits):

    """Execution of Metropolis, checking every single site 10 times."""

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        staple = staple_calculus(t, x, y, z, mu, U)

                        for _ in range(hits):

                            old_link = U[t, x, y, z, mu].copy()
                            S_old = calculate_S(old_link, staple, beta)

                            su3matrix = SU3SingleMatrix()
                            new_link = np.dot(su3matrix, old_link)
                            S_new = calculate_S(new_link, staple, beta)
                            dS = S_new - S_old

                            if dS < 0:
                                U[t, x, y, z, mu] = new_link
                            else:
                                if np.exp(-dS) > np.random.uniform(0, 1):
                                    U[t, x, y, z, mu] = new_link
                                else:
                                    U[t, x, y, z, mu] = old_link
    return U


@njit()
def diagonalization(Matrix):

    _, eVec = np.linalg.eig(Matrix)
    V = eVec
    return V


if __name__ == "__main__":

    import time

    measures = 20
    idecorrel = par.idecorrel

    R = 1
    T = 1
    beta_vec = (np.linspace(0.1, 10, 20)).tolist()
    U = initialize_lattice(1, N)

    # which kind of link update would you like to use?
    overrelax = True
    metropolis = False
    heatbath = False

    Smean = []
    Smean2 = []

    staple_start = np.zeros((su3, su3), complex)
    quat = np.zeros((su2, su2), complex)

    start = time.time()

    for beth in beta_vec:

        print(
            "exe for beta = ",
            round(beth, 2),
            "step",
            beta_vec.index(beth),
            "/",
            len(beta_vec),
        )
        obs = []
        obsame = []

        for _ in range(measures):

            if heatbath:
                U = HB_updating_links(beth, U, N)

            if metropolis:
                U = Metropolis(U, beth, hits=10)

            if overrelax:
                for _ in range(1):
                    U = OverRelaxation_(U, N)

            # two different wilson loops
            temp = WilsonAction(R, T, U)
            temp2 = WilsonAction(3, 3, U)

            print(temp)

            obs.append(temp)
            obsame.append(temp2)
        Smean.append(np.mean(obs))
        Smean2.append(np.mean(obsame))

    print(f"Execution time: {round(time.time() - start, 2)} s")
    plt.figure()
    plt.title(
        f"Average action for N = {N}, measures = {measures} for beta in [{min(beta_vec)},{max(beta_vec)}]",
        fontsize=17,
    )
    plt.plot(beta_vec, Smean, "go")
    plt.plot(beta_vec, Smean2, "bo")
    plt.xlabel(r"$\beta$", fontsize=15)
    plt.ylabel(r"<$S_{W_{11}}$>", fontsize=15)
    plt.legend(["W11", "W22"])
    plt.show()
