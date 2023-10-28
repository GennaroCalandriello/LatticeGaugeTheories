import numpy as np
from numba import njit

from functions import *
from HB_OR_SU3 import *


def connected_correlation_time(x: np.array, normalized=False):
    x_mean = x.mean()
    C_fixed = (x * x).mean() - x_mean ** 2
    max_time = int(len(x) / 20)
    C = np.array(
        [(C_fixed + (x[:-k] * x[k:]).mean() - x_mean ** 2) for k in range(1, max_time)]
    )
    if normalized == False:
        if C[0] != 0:
            return C / C[0]
        else:
            print("C0 è zero ")
        return C


if __name__ == "__main__":

    U = initialize_lattice(1, N)

    beta = 5.7
    msrs = 2000
    obs = []

    for i in range(msrs):

        print("measura", i)

        U = Metropolis(U, beta, hits=7)
        U = HB_updating_links(beta, U, N=5)

        obs.append(WilsonAction(1, 1, U))

    obs = np.array(obs)
    Corr = connected_correlation_time(obs)
    plt.figure()
    plt.plot(range(len(Corr)), Corr, "go")
    plt.show()

