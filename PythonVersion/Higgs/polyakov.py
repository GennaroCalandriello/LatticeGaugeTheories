import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from functions import *
from su2Higgs import *


@njit()
def Polyakov(U, m, n, beta):

    loop = np.array(((1 + 1j, 1 + 1j), (1 + 1j, 1 + 1j)))

    for t in range(N):
        # for mu in range(4):
        loop = loop @ U[m[0], m[1], m[2], t, 4]

    Poly = np.trace(loop)

    return beta ** N * Poly / (N ** 3)


@njit()
def Polyakov2(U, beta):

    """Sum of product of Polyakov loop, it's an order parameter for the phase transition in pure gauge theory
    from the confinement to the deconfinement phase. The Polyakov loop is related to the quark-antiquark 
    potential. For definition see Gattringer and reference 1."""

    somma = 0

    for z in range(N):
        for x in range(N):
            for y in range(N):
                loop = np.identity(su2) + 0j
                for t in range(N):
                    loop = loop @ U[z, x, y, t, 3]  # product only in one mu-direction

                somma += 1 / 3 * np.trace(loop)

    return somma / N ** 3


if __name__ == "__main__":

    U, _ = initialize_fields(1)

    betavec = np.linspace(2.0, 6, 15).tolist()
    measures = 10
    m = [1, 1, 1]

    polyakovloop = []

    for beta in betavec:
        obs = []
        print(f"exe for beta = {beta}, remaining: {len(betavec)-betavec.index(beta)}")

        for i in range(measures):
            
            U, _ = HeatBath_update(
                U, _, beta, 0
            )  # with k=0 ht heatbath reproduces the distribution of pure gauge SU(2)

            # Poly = Polyakov(U, m, 0, beta)
            Poly = Polyakov2(U, beta)
            obs.append((Poly))
            print((Poly))

        polyakovloop.append(np.mean(np.abs(obs)))

    np.savetxt(f"polyakovBare_N_{N}.txt", polyakovloop)

    plt.figure()
    plt.title(
        r"Polyakov loop  $L_{bare}=\frac{1}{N^3} \sum_{x} \frac{1}{N_c} Tr \hspace{0.3} \prod_{t} \hspace{0.3} U_4(x, t)$",
        fontsize=18,
    )
    plt.xlabel(r"$\beta$", fontsize=15)
    plt.ylabel("<|L|>", fontsize=15)
    plt.plot(betavec, polyakovloop, "ro")
    plt.legend(["N=7"])
    plt.show()
