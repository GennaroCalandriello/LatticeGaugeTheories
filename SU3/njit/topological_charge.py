# from hmc import *
from functools import partial
from matplotlib.pyplot import hist
from functions import *
from HB_OR_SU3 import *
import time


@njit(
    complex128[:, :](
        complex128[:, :, :, :, :, :, :], int64, int64, int64, int64, int64, int64
    )
)
def Plaq(U, x, y, z, t, mu, nu):

    a_mu = [0, 0, 0, 0]
    a_nu = [0, 0, 0, 0]
    a_mu[mu] = 1
    a_nu[nu] = 1
    P = np.zeros((3, 3), dtype=complex128)
    P = (
        U[x, y, z, t, mu]
        @ U[
            (x + a_mu[mu]) % Ns,
            (y + a_mu[mu]) % Ns,
            (z + a_mu[mu]) % Ns,
            (t + a_mu[mu]) % Nt,
            nu,
        ]
        @ U[
            (x + a_nu[nu]) % Ns,
            (y + a_nu[nu]) % Ns,
            (z + a_nu[nu]) % Ns,
            (t + a_nu[nu]) % Nt,
            mu,
        ]
        .conj()
        .T
        @ U[(x), (y), (z), (t), nu].conj().T
    )
    return P


@njit()
def epsilon_():
    epsilon = np.zeros((4, 4, 4, 4), dtype=int64)
    epsilon[0, 1, 2, 3] = epsilon[1, 2, 3, 0] = epsilon[2, 3, 0, 1] = epsilon[
        3, 0, 1, 2
    ] = 1
    epsilon[3, 2, 1, 0] = epsilon[2, 1, 0, 3] = epsilon[1, 0, 3, 2] = epsilon[
        0, 3, 2, 1
    ] = -1
    return epsilon


epsilon = epsilon_()


@njit()
def q(U):

    N = -1 / (32 * np.pi**2)
    # epsilon = epsilon_()
    # topo = []
    Q = 0

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    topotemp = 0
                    for mu in range(4):
                        for nu in range(4):
                            for rho in range(4):
                                for sigma in range(4):
                                    if epsilon[mu, nu, rho, sigma] != 0:
                                        topotemp += (
                                            N
                                            * epsilon[mu, nu, rho, sigma]
                                            * np.trace(
                                                Plaq(U, x, y, z, t, mu, nu)
                                                @ Plaq(U, x, y, z, t, rho, sigma)
                                            )
                                        )

                    # print("topotemp", topotemp)
                    # topo.append(topotemp)
                    Q += topotemp
    # topo = np.array(topo)
    return Q


def topoCalculus(U, beta):
    "return array of topological charges for each configuration"
    print("Topolone beta:", beta)
    topo_charge = []

    for _ in range(N_conf):
        U = HB_updating_links(beta, U)
        # U = OverRelaxation_(U)
        Q = q(U)
        if _ % 10 == 0:
            print("conf:", _)

        topo_charge.append(Q)

    return np.array(topo_charge)


def FreedmanDiaconis(spacings):

    q1, q3 = np.percentile(spacings, [25, 75])
    iqr = q3 - q1
    n = len(spacings)

    bin_width = 2 * iqr / (n ** (1 / 3))
    data_range = spacings.max() - spacings.min()
    num_bins = int(np.ceil(data_range / bin_width))

    return num_bins


def main():
    topototal = []
    U = initialize_lattice(1)

    start = time.time()

    with multiprocessing.Pool(processes=len(beta_vec)) as pool:
        temp = partial(topoCalculus, U)
        topo_charges = np.array(pool.map(temp, beta_vec))
    print("topo_charge", topo_charges.shape)

    # np.savetxt("topo_charge_beta_ 3.4_5.7_7.9.txt", topo_charges)

    print("lunghezza", len(topo_charges[0]))
    plt.figure()
    for t in range(len(topo_charges)):
        plt.hist(
            topo_charges[t].imag,
            bins=FreedmanDiaconis(topo_charges[t].imag),
            histtype="step",
        )
    plt.show()

    print("time:", round(time.time() - start))
    np.savetxt("topo_charge_beta_ 3.4_5.7_7.9.txt", topo_charges)


if __name__ == "__main__":
    pass

    # main()
