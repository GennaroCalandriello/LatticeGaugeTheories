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
                            # print(np.linalg.det(Uprime))
                            Uprime = GramSchmidt(Uprime, True)
                            Uprime = checkSU3(Uprime)
                            # Uprime = normalize(Uprime)

                            U[t, x, y, z, mu] = Uprime

                        else:
                            U[t, x, y, z, mu] = SU3SingleMatrix()

    return U


@njit()
def OverRelaxation_update(U, N):

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

                        A = staple_calculus(t, x, y, z, mu, U)
                        a = np.sqrt((np.linalg.det(A)))

                        if a != 0:

                            # A = A / a
                            Adagger = A.conj().T
                            H = np.dot(Adagger, A) ** 0.5
                            O = A @ (1 / H)

                            O = GramSchmidt(
                                O, exe=False
                            )  # with exe=False the matrix is only normalized
                            det_O = np.linalg.det(O.conj().T).real

                            if round(det_O) != 0:

                                # if round(det_O) == 1:
                                #     I_alpha = (np.identity(su3) + 0j) * (1)
                                #     Otilde = np.dot(O, I_alpha)

                                # if round(det_O) == -1:
                                #     I_alpha = (np.identity(su3) + 0j) * (-1)
                                #     Otilde = np.dot(O, I_alpha)

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
        wbar = GramSchmidt(wbar, False)

        # if a != 0:
        xw = quaternion(sampleA(beta, a))
        xx = xw * wbar.conj().T

        return xx

    else:
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
                    # U[N - 1, x, y, z] = U[0, x, y, z]
                    # U[t, N - 1, y, z] = U[t, 0, y, z]
                    # U[t, x, N - 1, z] = U[t, x, 0, z]
                    # U[t, x, y, N - 1] = U[t, x, y, 0]

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
def Metropolis(U, beta, sweeps):

    """Execution of Metropolis, checking every single site 10 times. Non funziona bene ancora!"""

    for t in range(N):
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for mu in range(4):

                        staple = staple_calculus(t, x, y, z, mu, U)

                        for _ in range(sweeps):

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

    measures = par.measures
    idecorrel = par.idecorrel

    R = 1
    T = 1
    beta_vec = (np.linspace(0.1, 8, 100)).tolist()
    U = initialize_lattice(1, N)

    # which kind of link update would you like to use?
    overrelax = False
    metropolis = True
    heatbath = False

    Smean = []
    Smean2 = []

    staple_start = np.zeros((su3, su3), complex)
    quat = np.zeros((su2, su2), complex)

    start = time.time()

    for beth in beta_vec:

        print("exe for beta = ", round(beth, 2), "step", beta_vec.index(beth))
        obs = []
        obsame = []

        for _ in range(measures):

            if heatbath:
                U = HB_updating_links(beth, U, N)

            if metropolis:
                U = Metropolis(U, beth, sweeps=10)

            if overrelax:
                for _ in range(idecorrel):
                    U = OverRelaxation_update(U, N)

            temp = WilsonAction(R, T, U)
            # U = Unew
            print(temp)

            obs.append(temp)
        Smean.append(np.mean(obs))
    print(f"Execution time: {round(time.time() - start, 2)} s")
    plt.figure()
    plt.title(
        f"Average action for N = {N}, measures = {measures} for beta in [{min(beta_vec)},{max(beta_vec)}]",
        fontsize=17,
    )
    plt.plot(beta_vec, Smean, "go")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"<$S_{W_{11}}$>")
    plt.legend()
    plt.show()

