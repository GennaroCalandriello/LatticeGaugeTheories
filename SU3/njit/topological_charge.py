# from hmc import *
from calendar import c
from matplotlib.pyplot import hist
from functions import *
from HB_OR_SU3 import *


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
        @ U[(x), (y), (z), (y), nu].conj().T
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


@njit(complex128(complex128[:, :, :, :, :, :, :]), fastmath=True)
def q(U):

    topo = 0
    N = 1 / (32 * np.pi**2)
    epsilon = epsilon_()
    print("esplsilonooo", epsilon[1, 2, 3, 0])
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        for nu in range(4):
                            for rho in range(4):
                                for sigma in range(4):
                                    topo += epsilon[mu, nu, rho, sigma] * np.trace(
                                        Plaq(U, x, y, z, t, mu, nu)
                                        @ Plaq(U, x, y, z, t, rho, sigma)
                                    )
    topo *= N
    return topo


if __name__ == "__main__":
    topo_charge = []
    U = initialize_lattice(1)
    for _ in range(400):
        if _ % 10 == 0:
            print("conf:", _)
        U = HB_updating_links(6.7, U)
        topo_charge.append(q(U))

    plt.figure()
    plt.hist(topo_charge, bins=100, density=True, histtype="step", color="red")
    plt.show()
