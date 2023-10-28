import dis
import numpy as np
from numba import njit
import multiprocessing
from functools import partial

from sympy import comp

from polyakov import *
from HB_OR_SU3 import *
from functions import *

path = "njit/datafiles/polyakov/static_potential/error/"
path1 = "njit/datafiles/polyakov/static_potential/"


@njit()
def WilsonLoop(R, T, U):

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
                            # a_nu = [0, 0, 0, 0]
                            # a_nu[nu] = 1

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
                                        (x + j * a_nu[0]) % Ns,
                                        (y + j * a_nu[1]) % Ns,
                                        (z + j * a_nu[2]) % Ns,
                                        (t + j * a_nu[3]) % Nt,
                                        nu,
                                    ]
                                    .conj()
                                    .T,
                                )

                            somma += np.trace(loop)

    return somma / (Ns**3 * Nt)


@njit()
def staticPot(data):
    """This is the data structure and ordered before passing in the function
    data = [[3, 0.4], [3, 0.2], [2, 0.3], [12, 0.5], [4, 0.1], [3, 0.2], [2, 1.2], [4, 2.3]]
    data.sort(key=lambda x: x[0])
    data = np.array(data)"""

    """njit works"""

    result = []
    sums = []
    counts = []

    for sublist in data:
        found = False
        for i, s in enumerate(sums):
            if s[0] == sublist[0]:
                s[1] += sublist[1]
                counts[i] += 1
                found = True
                break
        if not found:
            sums.append([sublist[0], sublist[1]])
            counts.append(1)

    mean_values = [[s[0], s[1] / counts[i]] for i, s in enumerate(sums)]
    return mean_values


@njit()
def StaticQuarkAntiquarkPotential2(U, beta):
    """Compute the static quark-antiquark potential V(r, T) for a given configuration U and beta, from
    Polyakov loops (ref. Gattringer)."""

    # Here set the number of configurations

    # Spatial points
    N_s = U.shape[2]

    V_ave = []

    for _ in range(N_conf):
        ensemble_average = []

        m = (0, 0, 0)  # Polyakov loop at the center of the lattice
        print("config ", _)

        U = HB_updating_links(beta, U, N_s)

        for u in range(1):
            U = Metropolis(U, beta, 5)
        # L0 = compute_polyakov_loop(U, [m[0], m[1], m[2]])
        for x in range(N_s):
            for y in range(N_s):
                for z in range(N_s):
                    L0 = compute_polyakov_loop(U, [x, y, z])

                    for dx in range(N_s):
                        for dy in range(N_s):
                            for dz in range(N_s):

                                r = np.sqrt(
                                    (x - dx) ** 2 + (y - dy) ** 2 + (z - dz) ** 2
                                )
                                Lr = compute_polyakov_loop(U, [dx, dy, dz])

                                trace_product = np.trace(L0.conj().T) * np.trace(Lr)
                                ensemble_average.append([r, trace_product])

        # Compute V_av(r, T)
        # --------------------------------------------------------------------
        tem = 0
        tem = staticPot(np.array(ensemble_average))
        # tem = np.array(tem)
        for t in tem:
            V_ave.append(t)

    V_final = staticPot(np.array(V_ave))
    # V_final = []
    # # for i in range(len(V_ave)):
    # #     media = np.mean(np.array(V_ave[i]))
    # #     media = np.log(media) * (-1 / 9)
    # #     V_final.append(media)

    # V_final = np.array(V_final)
    # V_ave = staticPot(np.array(ensemble_average))
    # V_ave = np.array(V_ave)
    # V_ave[:, 1] = -(1 / 9) * np.log(V_ave[:, 1])

    # V_ave.append(temp)
    # V_ave = np.array(V_ave)
    # print(V_ave.shape)

    return V_final  # V_ave, tem[:, 0]


def WilsonLoopStaticQuarkAntiquarkPotential(beta):

    R_arr = np.linspace(1, Ns, Ns).astype(int)
    V_arr = np.zeros((int(len(R_arr)), N_conf), dtype=np.complex128)
    U = initialize_lattice(1)

    for _ in range(thermalization):
        U = HB_updating_links(beta, U)
        print(f"thermalizing... {_}/{thermalization}")

    print("Lattice initialized", U.shape)

    # for n in range(10):
    #     print("thermalizing...")
    #     U = HB_updating_links(beta, U, N)

    #     for _ in range(10):
    #         U = Metropolis(U, beta, 5)

    for m in range(N_conf):
        print("config ", m)
        U = HB_updating_links(beta, U)
        U = OverRelaxation_(U)

        c = 0

        for R in R_arr:
            print("R ", R)
            W = WilsonLoop(R, 1, U)
            V_arr[c, m] = W
            c += 1
    np.savetxt(f"{path1}/V_Wilson_beta{beta}.txt", V_arr)
    V = -np.log(np.mean(V_arr[:, 200:], axis=1))
    print(V)

    plt.figure()
    plt.scatter(R_arr, V, s=10, c="r")
    plt.xlabel("R")
    plt.ylabel("V(R)")
    plt.title("Static potential")
    plt.show()


# @njit()
def get_key(x):
    return x[0]


# @njit()
def numba_sorted(data):

    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j][0] > data[j + 1][0]:
                data[j], data[j + 1] = data[j + 1], data[j]

    return data


@njit()
def group_data(data):
    data = [
        [3, 0.4],
        [3, 0.2],
        [2.1, 0.3],
        [12, 0.5],
        [4, 0.1],
        [3, 0.2],
        [2.1, 1.2],
        [4.33, 2.3],
        [4, 0.121],
    ]
    # Sort data by the first element
    # data.sort(key=lambda x: x[0])
    # data = np.array(data)
    # data = data[np.argsort(data[:, 0], kind="stable")]

    print(data)
    arr = data
    # arr = arr[np.argsort(arr[:, 0], kind="stable")]
    # data = numba_sorted(data)
    # data = sorted(data, key=lambda x: x[0])

    grouped_data = []
    temp = [data[0][0], [data[0][1]]]

    for i in range(1, len(data)):
        if data[i][0] == temp[0]:
            temp[1].append(data[i][1])
        else:
            grouped_data.append(temp)
            temp = [data[i][0], [data[i][1]]]

    grouped_data.append(temp)
    return grouped_data


########################################################POLYNEW#######################################################
def compute_polyakov(U):
    Ns = U.shape[0]
    Nt = U.shape[3]
    polyakov_loops = np.zeros((Ns, Ns, Ns), dtype=np.complex128)

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                loop = np.identity(3, dtype=np.complex128)
                for t in range(Nt):
                    loop = np.dot(loop, U[x, y, z, t, 3, :, :])
                polyakov_loops[x, y, z] = np.trace(loop)

    return polyakov_loops


def mainLoop():

    U = initialize_lattice(1)
    beta = 8.7
    # U = thermalize(U, beta)
    r_vecs = np.array(uniqueDistance())
    print(len(r_vecs))

    totalCorrelator = np.zeros((N_conf, len(r_vecs)), dtype=np.complex128)

    for _ in range(N_conf):
        print("config ", _)
        U = HB_updating_links(beta, U)
        U = OverRelaxation_(U)
        P = compute_polyakov(U)

        correlators = [compute_correlator(P, r_vec) for r_vec in r_vecs]
        correlators = np.array(correlators)

    corrPlot = []
    for j in range(Ns):
        corrPlot.append(-np.log(np.mean(totalCorrelator[:, j])))
    corrPlot = np.array(corrPlot)
    print(corrPlot)
    plt.figure()
    plt.scatter(range(Ns), corrPlot)
    plt.show()


def uniqueDistance():
    # Example value; you can change this to fit your case

    # Initialize an empty list to hold the tuples
    r = []

    # Initialize an empty set to hold the unique distances
    unique_distances = set()

    for i in range(Ns):
        for j in range(Ns):
            for k in range(Ns):
                # Skip the origin
                if i == 0 and j == 0 and k == 0:
                    continue

                # Calculate the distance
                distance = np.sqrt(i**2 + j**2 + k**2)

                # Check if this distance is unique
                if distance not in unique_distances:
                    # Add the distance to the set of unique distances
                    unique_distances.add(distance)

                    # Add the tuple to the list
                    r.append((i, j, k))
    rdist = []
    for i in range(len(r)):
        rdist.append(np.sqrt(r[i][0] ** 2 + r[i][1] ** 2 + r[i][2] ** 2))

    print(rdist)

    return r


###########################################################sono qua!!!


@njit()
def compute_correlator(polyakov_loops, r_vec):

    Ns = polyakov_loops.shape[0]
    correlator = 0.0

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):

                dx, dy, dz = r_vec
                correlator += polyakov_loops[x, y, z] * np.conjugate(
                    polyakov_loops[(x + dx) % Ns, (y + dy) % Ns, (z + dz) % Ns]
                )

    # Average over the entire lattice
    correlator /= Ns**3
    return np.abs(correlator)


if __name__ == "__main__":

    static1 = False
    static2 = False

    if static1:
        checkPath(path1)
        checkPath(path)

        # Assuming `configs` is a list of your 100 configurations
        U = initialize_lattice(1)
        V_average = StaticQuarkAntiquarkPotential2(U, beta_vec[1])
        V_average = np.array(V_average)
        V_average = V_average[np.argsort(V_average[:, 0], kind="stable")]
        V_average[:, 1] = -(1 / 9) * np.log(V_average[:, 1])
        print(V_average)

        print("V_average", V_average.shape)
        np.savetxt(f"{path1}/V_average_{beta_vec[1]}.txt", V_average)
        plt.figure()
        plt.scatter(V_average[:, 0], V_average[:, 1], s=10, c="r")
        plt.xlabel("r")
        plt.ylabel("V(r)")
        plt.title("Static potential")
        plt.show()

    if static2:

        WilsonLoopStaticQuarkAntiquarkPotential(4.5)

    mainLoop()
    uniqueDistance()
