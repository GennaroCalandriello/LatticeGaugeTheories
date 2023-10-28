# from hmc import *
from functools import partial
from functions import *
from HB_OR_SU3 import *
import time

"""[1] https://arxiv.org/pdf/1509.04259.pdf"""


@njit(
    complex128[:, :](
        complex128[:, :, :, :, :, :, :], int64, int64, int64, int64, int64, int64
    ),
    fastmath=True,
)
def Plaq(U, x, y, z, t, mu, nu):

    a_mu = [0, 0, 0, 0]
    a_nu = [0, 0, 0, 0]
    a_mu[mu] = 1
    a_nu[nu] = 1

    P = (
        U[
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
    ) + (
        U[
            (x + a_mu[0] - a_nu[0]) % Ns,
            (y + a_mu[1] - a_nu[1]) % Ns,
            (z + a_mu[2] - a_nu[2]) % Ns,
            (t + a_mu[3] - a_nu[3]) % Nt,
            nu,
        ]
        .conj()
        .T
        @ U[
            (x - a_nu[0]) % Ns,
            (y - a_nu[1]) % Ns,
            (z - a_nu[2]) % Ns,
            (t - a_nu[3]) % Nt,
            mu,
        ]
        .conj()
        .T
        @ U[
            (x - a_nu[0]) % Ns,
            (y - a_nu[1]) % Ns,
            (z - a_nu[2]) % Ns,
            (t - a_nu[3]) % Nt,
            nu,
        ]
    )

    P = U[x, y, z, t, mu] @ P
    return P


@njit(
    complex128[:, :](
        complex128[:, :, :, :, :, :, :], int64, int64, int64, int64, int64, int64
    ),
    fastmath=True,
    parallel=True,
)
def G_clover(U, x, y, z, t, mu, nu):
    """[1] eq. 14"""

    a_mu = [0, 0, 0, 0]
    a_nu = [0, 0, 0, 0]

    a_mu[mu] = 1
    a_nu[nu] = 1

    Gmunu = 0.25 * (
        U[x, y, z, t, mu]
        @ U[
            (x + a_mu[mu]) % Ns,
            (y + a_mu[mu]) % Ns,
            (z + a_mu[mu]) % Ns,
            (t + a_mu[mu]) % Nt,
            nu,
        ]
        @ U[
            (x + a_mu[mu]) % Ns,
            (y + a_mu[mu]) % Ns,
            (z + a_mu[mu]) % Ns,
            (t + a_mu[mu]) % Nt,
            mu,
        ]
        .conj()
        .T
        @ U[x, y, z, t, nu].conj().T
        + U[x, y, z, t, nu]
        @ U[
            (x - a_mu[mu] + a_nu[nu]) % Ns,
            (y - a_mu[mu] + a_nu[nu]) % Ns,
            (z - a_mu[mu] + a_nu[nu]) % Ns,
            (t - a_mu[mu] + a_nu[nu]) % Nt,
            mu,
        ]
        .conj()
        .T
        @ U[
            (x - a_mu[mu]) % Ns,
            (y - a_mu[mu]) % Ns,
            (z - a_mu[mu]) % Ns,
            (t - a_mu[mu]) % Nt,
            nu,
        ]
        .conj()
        .T
        @ U[
            (x - a_mu[mu]) % Ns,
            (y - a_mu[mu]) % Ns,
            (z - a_mu[mu]) % Ns,
            (t - a_mu[mu]) % Nt,
            mu,
        ]
        + U[
            (x - a_mu[mu]) % Ns,
            (y - a_mu[mu]) % Ns,
            (z - a_mu[mu]) % Ns,
            (t - a_mu[mu]) % Nt,
            mu,
        ]
        .conj()
        .T
        @ U[
            (x - a_mu[mu] - a_nu[nu]) % Ns,
            (y - a_mu[mu] - a_nu[nu]) % Ns,
            (z - a_mu[mu] - a_nu[nu]) % Ns,
            (t - a_mu[mu] - a_nu[nu]) % Nt,
            nu,
        ]
        .conj()
        .T
        @ U[
            (x - a_mu[mu] - a_nu[nu]) % Ns,
            (y - a_mu[mu] - a_nu[nu]) % Ns,
            (z - a_mu[mu] - a_nu[nu]) % Ns,
            (t - a_mu[mu] - a_nu[nu]) % Nt,
            mu,
        ]
        @ U[
            (x - a_nu[nu]) % Ns,
            (y - a_nu[nu]) % Ns,
            (z - a_nu[nu]) % Ns,
            (t - a_nu[nu]) % Nt,
            nu,
        ]
        + U[
            (x - a_nu[nu]) % Ns,
            (y - a_nu[nu]) % Ns,
            (z - a_nu[nu]) % Ns,
            (t - a_nu[nu]) % Nt,
            nu,
        ]
        .conj()
        .T
        @ U[
            (x - a_nu[nu]) % Ns,
            (y - a_nu[nu]) % Ns,
            (z - a_nu[nu]) % Ns,
            (t - a_nu[nu]) % Nt,
            mu,
        ]
        @ U[
            (x + a_mu[mu] - a_nu[nu]) % Ns,
            (y + a_mu[mu] - a_nu[nu]) % Ns,
            (z + a_mu[mu] - a_nu[nu]) % Ns,
            (t + a_mu[mu] - a_nu[nu]) % Nt,
            nu,
        ]
        @ U[x, y, z, t, mu].conj().T
    )


@njit(int64(int64, int64, int64, int64), fastmath=True)
def epsilon_(mu, nu, rho, sigma):

    eps = (
        np.sign(sigma - mu)
        * np.sign(rho - mu)
        * np.sign(nu - mu)
        * np.sign(sigma - nu)
        * np.sign(rho - nu)
        * np.sign(sigma - rho)
    )
    return eps


def levi_civita():
    epsilon = np.zeros((4, 4, 4, 4), dtype=int)

    # For even permutations (0,1,2,3)
    epsilon[0, 1, 2, 3] = 1
    epsilon[1, 2, 3, 0] = 1
    epsilon[2, 3, 0, 1] = 1
    epsilon[3, 0, 1, 2] = 1

    # For odd permutations (obtained by swapping indices)
    epsilon[3, 2, 1, 0] = -1
    epsilon[2, 1, 0, 3] = -1
    epsilon[1, 0, 3, 2] = -1
    epsilon[0, 3, 2, 1] = -1

    # Fill in other values by permuting these base values
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    if len(set([mu, nu, rho, sigma])) < 4:  # skip repeated indices
                        continue
                    perm = [mu, nu, rho, sigma]
                    epsilon[mu, nu, rho, sigma] = (
                        epsilon[0, 1, 2, 3]
                        * np.sign(
                            np.argsort(np.argsort(perm)).tolist().index(3)
                            - np.argsort(np.argsort(perm)).tolist().index(2)
                        )
                        * np.sign(
                            np.argsort(np.argsort(perm)).tolist().index(2)
                            - np.argsort(np.argsort(perm)).tolist().index(1)
                        )
                        * np.sign(
                            np.argsort(np.argsort(perm)).tolist().index(1)
                            - np.argsort(np.argsort(perm)).tolist().index(0)
                        )
                    )

    return epsilon


@njit(float64(complex128[:, :, :, :, :, :, :]), fastmath=True)
def q(U):

    N = -1 / (32 * np.pi**2)
    # epsilon = epsilon_()
    # topo = []
    Q = 0
    topotemp = 0
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    # topotemp = 0
                    for mu in range(4):
                        for nu in range(4):
                            for rho in range(4):
                                for sigma in range(4):
                                    if epsilon_(mu, nu, rho, sigma) != 0:
                                        topotemp += (
                                            N
                                            * epsilon_(mu, nu, rho, sigma)
                                            * np.trace(
                                                G_clover(U, x, y, z, t, mu, nu)
                                                @ G_clover(U, x, y, z, t, rho, sigma)
                                            )
                                        )

                    # print("topotemp", topotemp)
                    # topo.append(topotemp)
    Q = topotemp
    Q = Q.real
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
    # print(epsilon_(1, 0, 3, 2))
    # expilon = levi_civita()
    # print(expilon[1, 0, 3, 2])
    U = initialize_lattice(1)
    for _ in range(10):
        U = HB_updating_links(6.9, U)

        print(q(U))
