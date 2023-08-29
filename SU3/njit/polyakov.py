import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, float64, int64, complex128, int32
import ply

from functions import *
from HB_OR_SU3 import *
from parameters import *


import warnings
from numba.core.errors import NumbaPerformanceWarning

# Disable the performance warning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


Ns = Ns
Nt = Nt
bins = bins

# checkPath("njit/datafiles")

startingPath = f"njit/datafiles/polyakov"
path = f"{startingPath}/err"

pathStaticPot = f"{startingPath}/static_potential"
pathStaticPotErr = f"{startingPath}/static_potential/error"
pathChi = f"{startingPath}/susceptibility"
pathList = [path, pathStaticPot, pathStaticPotErr, pathChi]


@njit(parallel=True, cache=True, fastmath=True)
def Polyakov2(U, beta):

    """Sum of product of Polyakov loop, it's an order parameter for the phase transition in pure gauge theory
    from the confinement to the deconfinement phase. The Polyakov loop is related to the quark-antiquark
    potential. For definition see Gattringer and reference 1."""

    somma = 0

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                loop = np.identity(su3) + 0j
                for t in range(0, Nt):
                    # if t == N - 1:
                    #     U[x, y, z, t] = U[x, y, z, 0]
                    loop = (
                        loop @ U[x, y, z, t, 3]
                    )  # product only in one mu-direction solo per mu=4 ottengo qualcosa di apparentemente sensato

                somma += 1 / 3 * np.trace(loop)

    return somma / Ns**3


@njit(parallel=True, cache=True, fastmath=True)
def PolyakovLoopSum(U):

    Ns = U.shape[2]
    Nt = U.shape[3]
    """This function seems to be faster than Polyakov2"""

    # Initialize dimensions
    # Replace with the actual dimensions of your lattice

    # Initialize Polyakov loop to zero for each spatial point
    P_sum = 0
    # Loop through all spatial points
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                # Initialize a 3x3 identity matrix
                P = np.identity(3) + 0j

                # Loop through the time direction and multiply the link matrices
                for t in range(Nt):
                    P = np.dot(
                        P, U[x, y, z, t, 3]
                    )  # Assuming time direction is labeled by 3

                # Take the trace and normalize it by the dimension of the group
                # polyakov_loop[x, y, z] = np.trace(P) / 3.0
                P_sum += np.trace(P) / 3.0

    # average_polyakov_loop = np.mean(polyakov_loop)
    P_avg = P_sum / (Ns**3)
    return P_avg


##########-----------------Bootstrap------------------##########
@njit(fastmath=True)
def ricampionamento(array_osservabile, bin):
    sampler = []
    for _ in range(round(len(array_osservabile) / bin)):
        ii = np.random.randint(0, len(array_osservabile))
        sampler.extend(array_osservabile[ii : min(ii + bin, len(array_osservabile))])

    return np.array(sampler)


# @njit(complex128[:](complex128[:], float64), fastmath=True)
def bootstrap_(array_osservabile, bin, observable="mean"):

    print("Bin size: ", bin)
    array_osservabile = np.array(array_osservabile, dtype=np.float64)
    obs_array = []

    for _ in range(100):
        print(
            "resampling",
        )

        sample = ricampionamento(array_osservabile, bin)
        if observable == "chi":

            obs_array.append(calculate_susceptibility(sample, Nt, Ns))

        if observable == "mean":
            obs_array.append(np.mean(np.abs(sample)))

    sigma = np.std(obs_array)
    print("sigmaaaaaaaaaaaaaaaaaaa", sigma)
    return sigma


# -np.log((np.mean((values)))) / (9 * N_t)

# ----------------------------------------------------------------------------------------------------


def main3(beta):

    """This function calculates the averaged bare Polyakov loops for each spatial point and for each configuration."""

    U = initialize_lattice(1)
    print("exe for beta ", beta)
    obs = []

    for m in range(N_conf):
        if m % 50 == 0:
            print("measure ", m)
        U = HB_updating_links(beta, U)

        for _ in range(3):
            U = OverRelaxation_(U)

        temp = Polyakov2(U, beta)
        obs.append((temp))

    return obs


@njit(parallel=True, cache=True, fastmath=True)
def compute_polyakov_loop(U, x):
    """Compute the Polyakov loop for a given spatial point x and configuration U."""

    Nt = U.shape[3]
    loop = np.identity(su3) + 1j
    for t in range(Nt):
        loop = loop @ U[x[0], x[1], x[2], t, 3]

    return loop


# @njit(parallel=True, cache=True, fastmath=True)
def StaticQuarkAntiquarkPotential(U, beta):
    """Compute the static quark-antiquark potential V(r, T) for a given configuration U and beta, from
    Polyakov loops (ref. Gattringer)."""
    # Here set the number of configurations

    # Spatial points
    N_s = U.shape[2]
    ensemble_average = {}

    for _ in range(N_conf):
        m = (0, 0, 0)  # Polyakov loop at the center of the lattice
        print("config ", _)

        U = HB_updating_links(beta, U)
        NsCorr = 1  # Ns

        U = OverRelaxation_(U)

        # L0 = compute_polyakov_loop(U, [m[0], m[1], m[2]])
        for x in range(NsCorr):
            for y in range(NsCorr):
                for z in range(NsCorr):
                    L0 = compute_polyakov_loop(U, [x, y, z])

                    for dx in range(N_s):
                        for dy in range(N_s):
                            for dz in range(N_s):

                                r = np.sqrt(
                                    (x - dx) ** 2 + (y - dy) ** 2 + (z - dz) ** 2
                                )
                                Lr = compute_polyakov_loop(U, [dx, dy, dz])

                                if r not in ensemble_average:
                                    ensemble_average[r] = []

                                trace_product = np.trace(L0.conj().T) * np.trace(Lr)
                                ensemble_average[r].append((trace_product))

    # Compute V_av(r, T)
    # --------------------------------------------------------------------
    V_av = {}
    N_t = U.shape[0]
    dataForErrors = []
    # ensemble_average = list(ensemble_average.items())
    # ensemble_average.sort(key=lambda x: x[0])

    for r, values in ensemble_average.items():
        V_av[r] = -np.log((np.mean((values)))) / (9 * N_t)

        real_parts = [x.real for x in values]
        # imaginary_parts = [x.imag for x in values]

        # Now, you can save `real_parts` and `imaginary_parts` to a file
        dataForErrors.append([real_parts])

    V_av = list(V_av.items())
    V_av.sort(key=lambda x: x[0])
    V_av = np.array(V_av)
    print("vaveeeeee", V_av)
    np.savetxt(f"{pathStaticPot}/V_beta_{round(beta, 2)}", V_av)

    with open(f"{pathStaticPotErr}/dataError_beta{round(beta, 2)}.txt", "w") as file:
        for re in dataForErrors:
            file.write(f"{re}\n")
    # # --------------------------------------------------------------------
    print("scemo chi legge")


def checkPath(pathlista):

    for p in pathlista:
        p = str(p)
        print("path ", p)
        if os.path.exists(p) == False:
            os.makedirs(p)
        else:
            shutil.rmtree(p)
            os.makedirs(p)


def PolyakovLoopSusceptibility(beta):
    """This function produces the data, Polyakov loops, for the susceptibility calculation."""

    U = initialize_lattice(1)
    poly = []
    print("exe for beta ", beta)

    for _ in range(N_conf):
        print("measure ", _)
        U = HB_updating_links(beta, U)
        U = OverRelaxation_(U)
        temp = PolyakovLoopSum(U)
        poly.append(temp)

    np.savetxt(f"{pathChi}/PolyakovLoopsBeta_{round(beta, 2)}", poly)
    return poly


@njit(parallel=True, cache=True, fastmath=True)
def calculate_susceptibility(polyakov_loop_samples, N_t, N_s, kind=None):

    # Remove thermalization
    polyakov_loop_samples = polyakov_loop_samples
    V = N_t * N_s**3

    if kind == "imag":
        ImP = np.imag(polyakov_loop_samples)
        avg_polyakov = np.mean(ImP)
        avg_polyakov_squared = np.mean(ImP**2)
        chi_polyakov = V * (avg_polyakov_squared - (avg_polyakov**2))

        return chi_polyakov, avg_polyakov, avg_polyakov_squared

    if kind == None:

        avg_polyakov = np.mean(np.abs(polyakov_loop_samples))
        avg_polyakov = avg_polyakov
        avg_polyakov_squared = np.mean(np.abs(polyakov_loop_samples) ** 2)
        chi_polyakov = V * (avg_polyakov_squared - (avg_polyakov**2))
        # print("pizza, kiwi e", avg_polyakov**2)
        # print("squaqquerone", avg_polyakov_squared)

        return chi_polyakov, avg_polyakov, avg_polyakov_squared


# @njit(float64(complex128[:]), fastmath=True)
def errorCalc(data, beta):

    obs = []
    with multiprocessing.Pool(processes=len(bins)) as pool:
        print("Calculate for beta = ", round(beta, 2))
        part = partial(bootstrap_, data)
        obs = np.array(pool.map(part, bins))

    return max(obs)


def fitStaticPotential(r, a1, a2, a3):

    pass


def densityplot(data, beta):
    """Plot the density of the Polyakov loop for a given beta"""
    from scipy import stats

    m1 = data.real
    m2 = data.imag
    xmin = m1.min()
    xmax = m1.max()
    ymin = m2.min()
    ymax = m2.max()

    # get the density estimation
    X, Y = np.mgrid[xmin:xmax:400j, ymin:ymax:400j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # plot the result

    #
    plt.figure()
    plt.scatter(m1, m2, s=1, facecolor="white", alpha=0.13)
    plt.imshow(
        np.rot90(Z),
        extent=[xmin, xmax, ymin, ymax],
        cmap="jet",
        alpha=1,
    )
    plt.colorbar(label="Density")
    plt.xlabel(r"$Re(\langle P \rangle)$", fontsize=18)
    plt.ylabel(r"$Im(\langle P \rangle)$", fontsize=18)
    plt.title(r"Polyakov loops for $\beta = $" + f"{round(beta, 2)}", fontsize=22)
    plt.show()


def thermalize(U, beta, HB=True, Metro=False, OR=False):

    for _ in range(thermalization):
        print("thermalization {}/{}".format(_, thermalization))
        if HB == True:
            U = HB_updating_links(beta, U)
        if Metro == True:
            U = Metropolis(U, beta, 5)
        if OR == True:
            U = OverRelaxation_(U)
    return U


if __name__ == "__main__":

    import multiprocessing
    from functools import partial

    U = initialize_lattice(1)

    analysis = False
    staticpotential = True
    centersymmetry = False
    susceptibility = False
    plot = False

    if analysis:
        import re

        s = "Here are some numbers (1.23, 4.56) and (7.89, 0.12) with other text."

        # graphicPlot()
        # pass
        for b in beta_vec:
            data = []
            with open(
                f"{pathStaticPotErr}/dataError_beta{round(b, 2)}.txt", "r"
            ) as file:
                print("beta = ", b)
                for line in file:
                    # print(line)
                    # Find all float numbers in the string
                    float_numbers = re.findall(r"[-+]?[0-9]*\.[0-9]+", line)
                    data.append(float_numbers)

            error = []
            print("Calculate error for beta = ", b)
            count = 0
            for d in data:
                count += 1
                d = np.array(d, dtype=np.float64)
                error.append(errorCalc(d, b))
                print("distance value number: ", count)
            np.savetxt(f"{pathStaticPotErr}/error_beta_{round(b, 2)}", error)

    if centersymmetry:
        """Here we calculate the Polyakov loop to visualize the center symmetry breaking"""
        with multiprocessing.Pool(processes=len(beta_vec)) as pool:
            polyakov = np.array(pool.map(main3, beta_vec))
            pool.close()
            pool.join()

        c = 0
        for pol in polyakov:
            np.savetxt(
                f"{pathStaticPotErr}/polyek_beta{round(beta_vec[c], 2)}.txt", pol
            )

            imagina = pol.imag
            reale = pol.real

            print(pol)
            plt.figure()
            plt.plot(pol.real, pol.imag, "o", color="blue")
            plt.show()
            c += 1

    if susceptibility:
        exe = 0  # False for 0
        errorExe = 0

        if errorExe != 0:

            errori = []

            for b in beta_vec:
                print("beta = ", b)
                pol = np.loadtxt(
                    f"{pathChi}/polyakovLoopsBeta_{round(b, 2)}", dtype=complex
                )
                # remove thermalization
                pol = pol[200:]
                errori.append(errorCalc(pol, b))

            np.savetxt(f"{pathChi}/errorimean.txt", errori)

        if exe != 0:
            checkPath(pathList)
            with multiprocessing.Pool(processes=len(beta_vec)) as pool:
                susc = np.array(pool.map(PolyakovLoopSusceptibility, beta_vec))
                pool.close()
                pool.join()

        susceptibilities = []
        mean = []
        measquare = []

        for b in beta_vec:
            pol = np.loadtxt(
                f"{pathChi}/polyakovLoopsBeta_{round(b, 2)}", dtype=complex
            )

            # remove thermalization
            # pol = pol[200:]
            if round(b, 2) == 5.2 or round(b, 2) == 5.71 or round(b, 2) == 6.23:
                plt.figure(figsize=(8, 8))
                plt.title(
                    f"MC histories of Re(P) for " r"$\beta =$" f"{round(b, 2)}",
                    fontsize=18,
                )
                plt.xlabel("steps", fontsize=18)
                plt.ylabel(r"$Re(P)$", fontsize=18)
                plt.plot(range(len(pol)), pol.real, "+", color="blue")
                plt.show()

            s, m, ms = calculate_susceptibility(pol, Nt, Ns)
            susceptibilities.append(s)
            mean.append(m)
            measquare.append(ms)

        errors = np.loadtxt(f"{pathChi}/errorimean.txt") * 9

        plt.errorbar(
            beta_vec,
            mean,
            yerr=errors,
            fmt="o",
            color="red",
            barsabove=True,
            capsize=3,
            elinewidth=1,
            markeredgewidth=1,
            ecolor="blue",
        )
        plt.xlabel(r"$\beta$", fontsize=18)
        plt.ylabel(r"$\langle |P| \rangle$", fontsize=18)
        plt.title("Polyakov Loop", fontsize=20)
        plt.show()

    if plot:
        for b in beta_vec:

            P = np.loadtxt(
                f"{pathStaticPotErr}/polyek_beta{round(b, 2)}.txt", dtype=complex
            )
            densityplot(P, b)

    if staticpotential:
        checkPath([pathStaticPot, pathStaticPotErr])

        U = initialize_lattice(1)
        U = thermalize(U, 5.2, HB=True, Metro=False, OR=False)
        StaticQuarkAntiquarkPotential(U, 6.8)
        pass

    v = np.loadtxt(f"{pathStaticPot}/V_beta_{round(6.8, 2)}", dtype=complex)
    R = v[:, 0]
    ave = v[:, 1]
    plt.plot(R, ave, "o", color="blue")
    plt.show()
