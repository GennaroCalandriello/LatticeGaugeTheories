import numpy as np
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt

from statistics import *
from randomSU2 import *
from su3_heatbath_overrelaxation import *
from algebra import *
import parameters as par


def main_exe(beta):

    measures = 50
    N = 5
    idecorrel = 4
    print(f"Executing for beta = {beta}")

    U = initialize_lattice(1, N)
    staple = np.zeros((su3, su3), complex)

    overrelax = True
    obs = []

    for m in range(measures):
        if overrelax:
            for _ in range(idecorrel):
                U = OverRelaxation_update(U, N, staple)

        U = updating_links(beta, U, N, staple)
        temp = WilsonAction(1, 1, U)
        print(f"Beta = {round(beta, 1)}, measure = {m}--{round(temp, 3)}")

        if m >= 20:
            obs.append(temp)

    S_mean = np.mean(obs)

    return S_mean


if __name__ == "__main__":

    beta_vec = np.linspace(2.0, 10, 20)

    with multiprocessing.Pool(processes=len(beta_vec)) as pool:
        results = np.array(pool.map(main_exe, beta_vec), dtype="object")
        pool.close()
        pool.join()

    plt.figure()
    plt.plot(beta_vec, results, "go--")
    plt.show()
