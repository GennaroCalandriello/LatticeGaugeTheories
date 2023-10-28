import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial

from functions import *
from su2Higgs import *


def main(par, kvec, measures, beta):

    R, T = 1, 1
    """par specify which parameter do you want to vary during simulation. This function calculates
    S_Wilson+S_Higgs varying beta, at fixed k array for each process"""

    U, phi = initialize_fields(1)

    betavec = np.linspace(0.1, 8.0, 25).tolist()

    results = []

    if par == "beta":

        k = 0.8

        beta = 0

        for beta in betavec:

            print(
                f"exe for beta = {beta}, remaining {len(betavec)-betavec.index(beta)}"
            )

            obs = []

            for i in range(measures):

                U = HeatBath_update(U, phi, beta, k)
                phi = Updating_Higgs(U, phi, beta, k)

                temp = WilsonHiggsAction(R, T, U, phi, k)
                print(temp)
                obs.append(temp)

            results.append(np.mean(obs).real)
            print("mean", np.mean(obs))

    if par == "k":

        print("execution for beta = ", beta)

        # k = 0.0

        for k in kvec:

            print(
                f"exe for k = {round(k, 2)}, for beta ={beta} remaining {len(kvec)-kvec.index(k)} measures"
            )

            obs = []

            for i in range(measures):

                U = HeatBath_update(U, phi, beta, k)
                phi = Updating_Higgs(U, phi, beta, k)

                temp = WilsonHiggsAction(R, T, U, phi, k)
                # temp = extendedHiggsAction(R, T, U, phi, k)
                print(round(temp.real, 4))
                obs.append(temp)

            results.append(np.mean(obs).real)

    return results


if __name__ == "__main__":

    import time

    start = time.time()

    betavec = np.linspace(0.4, 4.5, 10).tolist()
    kvec = np.linspace(-2.2, 2.2, 100).tolist()
    par = "k"
    measures = 30

    with multiprocessing.Pool(processes=len(betavec)) as pool:
        part = partial(main, par, kvec, measures)
        results = np.array(pool.map(part, betavec))
        pool.close()
        pool.join()

    print("Execution time: ", round(time.time() - start, 3), "s")

    plt.figure()
    plt.title("Higgs behavior at various beta", fontsize=20)

    for b in range(len(results)):
        plt.scatter(kvec, results[b], s=8.0, label=round(betavec[b], 2))

    np.savetxt("Higgs_various_beta.txt", results)
    np.savetxt("beta_array.txt", betavec)
    plt.xlabel("k", fontsize=15)
    plt.ylabel("<S>", fontsize=15)
    plt.legend()
    plt.show()

