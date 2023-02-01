import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from functions import *
from HB_OR_SU3 import *

measures = 5000


def Polyakovmacomecazz(U):

    trace1 = []
    for x in range(N):
        for y in range(N):
            for z in range(N):
                poly1 = np.identity(su3) + 0j
                for t in range(N):
                    poly1 = poly1 @ U[x, y, z, t, 3]

                trace1.append(np.trace(poly1))

    return np.mean(np.array(trace1))


def Polyakov3(U):

    trace1 = []
    a1, a2, a3 = 0, 0, 1
    poly1 = np.identity(su3) + 0j
    for t in range(N):
        poly1 = poly1 @ U[a1, a2, a3, t, 3]

    trace1 = np.trace(poly1) / su3

    return trace1


@njit()
def Polyakov2(U, beta):

    """Sum of product of Polyakov loop, it's an order parameter for the phase transition in pure gauge theory
    from the confinement to the deconfinement phase. The Polyakov loop is related to the quark-antiquark 
    potential. For definition see Gattringer and reference 1."""

    somma = 0

    for x in range(N):
        for y in range(N):
            for z in range(N):
                loop = np.identity(su3) + 0j
                for t in range(0, N):
                    # if t == N - 1:
                    #     U[x, y, z, t] = U[x, y, z, 0]
                    loop = (
                        loop @ U[x, y, z, t, 3]
                    )  # product only in one mu-direction solo per mu=4 ottengo qualcosa di apparentemente sensato

                somma += 1 / 3 * np.trace(loop)

    return somma / N ** 3


@njit()
def Polyakov2punto1(U, beta):

    """Sum of product of Polyakov loop, it's an order parameter for the phase transition in pure gauge theory
    from the confinement to the deconfinement phase. The Polyakov loop is related to the quark-antiquark 
    potential. For definition see Gattringer and reference 1."""

    somma = 0
    sommavec = []

    for x in range(N):
        for y in range(N):
            for z in range(N):
                loop = np.identity(su3) + 0j
                for t in range(0, N):
                    # U[x, y, z, t] = PeriodicBC(U, x, y, z, t, N)
                    loop = loop @ U[x, y, z, t, 3]  # product only in one mu-direction

                somma += 1 / 3 * np.trace(loop)
                sommavec.append(somma)
    sommavec = np.array(sommavec) / N ** 3

    return np.mean(sommavec)


def main3(U, beta):

    obs = []
    print("exe for beta =", beta)

    for m in range(
        measures
    ):  # qui ci sono le misure per ogni distanza su cui fare il valor medio
        print("measure", m)

        U = HB_updating_links(beta, U, N)  # faccio l'update -> cambio cammino

        # temp = Polyakovmacomecazz(U)
        temp = Polyakov3(U)
        # temp = Polyakov2(U, beta)
        obs.append((temp))

    return np.array(obs)


def main2(U, beta):
    print("exe for beta ", beta)
    obs = []
    for m in range(measures):
        if m % 20 == 0:
            print("measure ", m)
        U = HB_updating_links(beta, U, N)
        # U = Metropolis(U, beta, 5)
        temp = Polyakov2(U, beta)
        obs.append(abs(temp))

    return np.mean(obs)


def main2punto1(U, beta):
    print("exe for beta ", beta)
    obs = []
    for m in range(measures):
        print("measure ", m)
        U = HB_updating_links(beta, U, N)

        temp = Polyakov2punto1(U, beta)
        obs.append((temp))

    return np.mean(np.array(obs))


if __name__ == "__main__":

    import multiprocessing
    from functools import partial

    U = initialize_lattice(1, N)
    betavec = np.linspace(5, 9, 15).tolist()

    polyak1 = False
    polyak2 = True
    polyak3 = False

    if polyak3:
        # trace of temporal product of link variables
        with multiprocessing.Pool(processes=len(betavec)) as pool:
            part = partial(main3, U)
            results = np.array(pool.map(part, betavec))
            pool.close()
            pool.join()

        plt.figure()

        for i in range(len(betavec)):
            plt.plot(
                results[i].real,
                results[i].imag,
                "o",
                label=f"beta {round(betavec[i], 2)}",
            )

        plt.title(r"Polyakov Loop vari beta")
        plt.xlabel("Re{P}", fontsize=15)
        plt.ylabel("Im{P}", fontsize=15)
        plt.legend()
        plt.show()

    if polyak2:
        # spatial average of sum of trace of product
        with multiprocessing.Pool(processes=len(betavec)) as pool:
            part = partial(main2, U)
            results = np.array(pool.map(part, betavec))
            pool.close()
            pool.join()

        plt.title(r"Polyakov sum vari beta")
        plt.plot(betavec, (results), "o")
        plt.ylabel(r"$<|P|>$", fontsize=15)
        plt.xlabel(r"$\beta$", fontsize=15)
        plt.legend()
        plt.show()

