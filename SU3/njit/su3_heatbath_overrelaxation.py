import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit

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
def updating_links(beta, U, N, staple_start):

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
                        staple_start = np.array(
                            (
                                (0 + 0j, 0 + 0j, 0 + 0j),
                                (0 + 0j, 0 + 0j, 0 + 0j),
                                (0 + 0j, 0 + 0j, 0 + 0j),
                            )
                        )

                        A = staple_calculus(t, x, y, z, mu, U, staple_start)
                        W = np.dot(Utemp, A)

                        a = det_calculus(W)

                        if a != 0:
                            # else:
                            r = heatbath_SU3(W, beta)
                            R = np.array(
                                (
                                    (r[0, 0], r[0, 1], 0),
                                    (r[1, 0], r[1, 1], 0),
                                    (0, 0, 1),
                                )
                            )

                            W = np.dot(R, W)
                            s = heatbath_SU3(W, beta)
                            S = np.array(
                                (
                                    (s[0, 0], 0, s[0, 1]),
                                    (0, 1, 0),
                                    (s[1, 0], 0, s[1, 1]),
                                )
                            )
                            W = np.dot(S, W)

                            tt = heatbath_SU3(W, beta)
                            T = np.array(
                                (
                                    (1, 0, 0),
                                    (0, tt[0, 0], tt[0, 1]),
                                    (0, tt[1, 0], tt[1, 1]),
                                )
                            )
                            # U'=TSRU

                            Uprime = np.dot(T, np.dot(S, np.dot(R, Utemp)))
                            Uprime = GramSchmidt(Uprime, True)

                            U[t, x, y, z, mu] = Uprime

                        else:
                            continue

    return U


@njit()
def OverRelaxation_update(U, N, staple):

    """Ettore Vicari, An overrelaxed Monte Carlo algorithm
        for SU(3) lattice gauge theories"""

    # I don't know if it works good!
    somma = 0
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for mu in range(4):

                        Utemp = U[t, x, y, z, mu]

                        staple = np.array(
                            (
                                (0 + 0j, 0 + 0j, 0 + 0j),
                                (0 + 0j, 0 + 0j, 0 + 0j),
                                (0 + 0j, 0 + 0j, 0 + 0j),
                            )
                        )

                        A = staple_calculus(t, x, y, z, mu, U, staple)
                        a = np.sqrt((np.linalg.det(A)))

                        if a != 0:

                            A = A / a
                            Adagger = A.conj().T
                            H = np.dot(Adagger, A) ** 0.5
                            O = A @ (1 / H)

                            # with exe=False the matrix is only normalized
                            O = GramSchmidt(O, False)
                            det_O = np.linalg.det(O.conj().T).real

                            if round(det_O) != 0:

                                if round(det_O) == 1:
                                    I_alpha = (np.identity(su3) + 0j) * (1)
                                    Otilde = np.dot(O, I_alpha)

                                if round(det_O) == -1:
                                    I_alpha = (np.identity(su3) + 0j) * (-1)
                                    Otilde = np.dot(O, I_alpha)

                                V = diagonalization(H)

                                Uprime = np.dot(V, np.dot(Utemp, np.dot(O, V.conj().T)))

                                reflection = np.random.randint(1, 4)

                                Uprime = reflectionSU3(Uprime, reflection)
                                Ufinal = np.dot(
                                    V.conj().T, np.dot(Uprime, np.dot(V, O.conj().T)),
                                )

                                detU = np.linalg.det(Ufinal).real

                                if round(detU) == 1:
                                    I_alpha = (np.identity(su3) + 0j) * (1)
                                    Ufinal = np.dot(Ufinal, I_alpha)

                                if round(detU) == -1:
                                    I_alpha = (np.identity(su3) + 0j) * (-1)
                                    Ufinal = np.dot(Ufinal, I_alpha)

                                U[t, x, y, z, mu] = Ufinal

    return U


@njit()
def heatbath_SU3(W, beta):

    """Construct the heatbath update for the link variable, see Gattringer for some details"""

    # print(W)
    w = getA(W)

    a = np.sqrt(np.abs(det_calculus(W)))

    if a != 0:
        wbar = quaternion(w)

        # if a != 0:
        xw = quaternion(sampleA(beta, a))
        xx = xw * wbar.conj().T

        return xx

    else:
        print("è zèro!")
        r0 = np.random.uniform(-0.5, 0.5)
        x0 = np.sign(r0) * np.sqrt(1 - epsilon ** 2)

        r = np.random.random((3)) - 0.5
        x = epsilon * r / np.linalg.norm(r)

        rmatrix = x0 * np.identity(2) + 1j * x[0] * sx + 1j * x[1] * sy + 1j * x[2] * sz

        return rmatrix


@njit()
def WilsonAction(R, T, U):

    """Name says"""

    somma = 0
    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):

                    # U[t, x, y, z] = PeriodicBC(U, t, x, y, z, N)

                    for mu in range(4):

                        a_mu = [0, 0, 0, 0]
                        a_mu[mu] = 1

                        nu = 0
                        # while nu < mu:
                        for nu in range(4):
                            # i = 0
                            # j = 0
                            if nu != mu:

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
def diagonalization(Matrix):

    eVal, eVec = np.linalg.eig(Matrix)
    V = eVec
    return V


if __name__ == "__main__":

    measures = 80
    idecorrel = 4
    R = 1
    T = 1
    beta_vec = np.linspace(1.0, 10, 50)  # np.linspace(3.0, 7, 10)
    U = initialize_lattice(1, N)
    overrelax = True

    Smean = []
    Smean2 = []

    staple_start = np.zeros((su3, su3), complex)
    quat = np.zeros((su2, su2), complex)

    for beth in beta_vec:

        print("exe for beta = ", beth)
        obs = []
        obsame = []

        for _ in range(measures):
            U = updating_links(beth, U, N, staple_start)

            if overrelax:
                for _ in range(idecorrel):
                    U = OverRelaxation_update(U, N, staple_start)

            temp = WilsonAction(R, T, U)

            print(temp)

            obs.append(temp)
        Smean.append(np.mean(obs))

    plt.figure()
    plt.title(
        f"Average action for N = {N}, measures = {measures} for beta in [{min(beta_vec)},{max(beta_vec)}]",
        fontsize=17,
    )
    plt.plot(beta_vec, Smean, "go--")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"<$S_{W_{11}}$>")
    plt.legend()
    plt.show()

