from math import exp
from topological_charge import *
from parameters import *

"""references: 
[1] https://arxiv.org/pdf/1509.04259.pdf"""


@njit(
    complex128[:, :](
        complex128[:, :, :, :, :, :, :], int64, int64, int64, int64, int64
    ),
    fastmath=True,
    parallel=True,
)
def Xrect(U, x, y, z, t, mu):

    """Rectangular Wilson loop
    basic idea to calculate the force:
    Xmu = c0*Xmu(plaq)+c1*Xmu(rect)
    with c1 = -1/12 and c0+8c1 = 1

    see [1]"""

    a_mu = [0, 0, 0, 0]
    a_mu[mu] = 1
    Xmu = np.zeros((su3, su3), dtype=complex128)
    for nu in range(4):

        a_nu = [0, 0, 0, 0]
        a_nu[nu] = 1

        if nu != mu:
            Xmu += (
                (
                    (
                        U[x, y, z, t, nu]
                        @ U[
                            (x + a_nu[nu]) % Ns,
                            (y + a_nu[nu]) % Ns,
                            (z + a_nu[nu]) % Ns,
                            (t + a_nu[nu]) % Nt,
                            nu,
                        ]
                        @ U[
                            (x + 2 * a_nu[nu]) % Ns,
                            (y + 2 * a_nu[nu]) % Ns,
                            (z + 2 * a_nu[nu]) % Ns,
                            (t + 2 * a_nu[nu]) % Nt,
                            mu,
                        ]
                        @ U[
                            (x + a_nu[nu] + a_mu[mu]) % Ns,
                            (y + a_nu[nu] + a_mu[mu]) % Ns,
                            (z + a_nu[nu] + a_mu[mu]) % Ns,
                            (t + a_nu[nu] + a_mu[mu]) % Nt,
                            nu,
                        ]
                        .conj()
                        .T
                        @ U[
                            (x + a_mu[mu]) % Ns,
                            (y + a_mu[mu]) % Ns,
                            (z + a_mu[mu]) % Ns,
                            (t + a_mu[mu]) % Nt,
                            nu,
                        ]
                        .conj()
                        .T
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
                            (x - 2 * a_nu[nu]) % Ns,
                            (y - 2 * a_nu[nu]) % Ns,
                            (z - 2 * a_nu[nu]) % Ns,
                            (t - 2 * a_nu[nu]) % Nt,
                            nu,
                        ]
                        .conj()
                        .T
                        @ U[
                            (x - 2 * a_nu[nu]) % Ns,
                            (y - 2 * a_nu[nu]) % Ns,
                            (z - 2 * a_nu[nu]) % Ns,
                            (t - 2 * a_nu[nu]) % Nt,
                            mu,
                        ]
                        @ U[
                            (x - 2 * a_nu[nu] + a_mu[mu]) % Ns,
                            (y - 2 * a_nu[nu] + a_mu[mu]) % Ns,
                            (z - 2 * a_nu[nu] + a_mu[mu]) % Ns,
                            (t - 2 * a_nu[nu] + a_mu[mu]) % Nt,
                            nu,
                        ]
                        @ U[
                            (x - a_nu[nu] + a_mu[mu]) % Ns,
                            (y - a_nu[nu] + a_mu[mu]) % Ns,
                            (z - a_nu[nu] + a_mu[mu]) % Ns,
                            (t - a_nu[nu] + a_mu[mu]) % Nt,
                            nu,
                        ]
                    )
                    + U[
                        (x) % Ns,
                        (y) % Ns,
                        (z) % Ns,
                        (t) % Nt,
                        nu,
                    ]
                    @ U[
                        (x + a_nu[nu]) % Ns,
                        (y + a_nu[nu]) % Ns,
                        (z + a_nu[nu]) % Ns,
                        (t + a_nu[nu]) % Nt,
                        mu,
                    ]
                    @ U[
                        (x + a_nu[nu] + a_mu[mu]) % Ns,
                        (y + a_nu[nu] + a_mu[mu]) % Ns,
                        (z + a_nu[nu] + a_mu[mu]) % Ns,
                        (t + a_nu[nu] + a_mu[mu]) % Nt,
                        mu,
                    ]
                    @ U[
                        (x + 2 * a_mu[mu]) % Ns,
                        (y + 2 * a_mu[mu]) % Ns,
                        (z + 2 * a_mu[mu]) % Ns,
                        (t + 2 * a_mu[mu]) % Nt,
                        nu,
                    ]
                    .conj()
                    .T
                    @ U[
                        (x + a_mu[mu]) % Ns,
                        (y + a_mu[mu]) % Ns,
                        (z + a_mu[mu]) % Ns,
                        (t + a_mu[mu]) % Nt,
                        mu,
                    ]
                    .conj()
                    .T
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
                        (x - a_nu[nu] + a_mu[mu]) % Ns,
                        (y - a_nu[nu] + a_mu[mu]) % Ns,
                        (z - a_nu[nu] + a_mu[mu]) % Ns,
                        (t - a_nu[nu] + a_mu[mu]) % Nt,
                        mu,
                    ]
                    @ U[
                        (x - a_nu[nu] + 2 * a_mu[mu]) % Ns,
                        (y - a_nu[nu] + 2 * a_mu[mu]) % Ns,
                        (z - a_nu[nu] + 2 * a_mu[mu]) % Ns,
                        (t - a_nu[nu] + 2 * a_mu[mu]) % Nt,
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
                )
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
                    (x - a_mu[mu]) % Ns,
                    (y - a_mu[mu]) % Ns,
                    (z - a_mu[mu]) % Ns,
                    (t - a_mu[mu]) % Nt,
                    nu,
                ]
                @ U[
                    (x - a_mu[mu] + a_nu[nu]) % Ns,
                    (y - a_mu[mu] + a_nu[nu]) % Ns,
                    (z - a_mu[mu] + a_nu[nu]) % Ns,
                    (t - a_mu[mu] + a_nu[nu]) % Nt,
                    mu,
                ]
                @ U[
                    (x + a_nu[nu]) % Ns,
                    (y + a_nu[nu]) % Ns,
                    (z + a_nu[nu]) % Ns,
                    (t + a_nu[nu]) % Nt,
                    mu,
                ]
                @ U[
                    (x + a_mu[mu]) % Ns,
                    (y + a_mu[mu]) % Ns,
                    (z + a_mu[mu]) % Ns,
                    (t + a_mu[mu]) % Nt,
                    nu,
                ]
                .conj()
                .T
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
                    (x - a_nu[nu] - a_mu[mu]) % Ns,
                    (y - a_nu[nu] - a_mu[mu]) % Ns,
                    (z - a_nu[nu] - a_mu[mu]) % Ns,
                    (t - a_nu[nu] - a_mu[mu]) % Nt,
                    nu,
                ]
                .conj()
                .T
                @ U[
                    (x - a_nu[nu] - a_mu[mu]) % Ns,
                    (y - a_nu[nu] - a_mu[mu]) % Ns,
                    (z - a_nu[nu] - a_mu[mu]) % Ns,
                    (t - a_nu[nu] - a_mu[mu]) % Nt,
                    mu,
                ]
                @ U[
                    (x - a_nu[nu]) % Ns,
                    (y - a_nu[nu]) % Ns,
                    (z - a_nu[nu]) % Ns,
                    (t - a_nu[nu]) % Nt,
                    mu,
                ]
                @ U[
                    (x - a_nu[nu] + a_mu[mu]) % Ns,
                    (y - a_nu[nu] + a_mu[mu]) % Ns,
                    (z - a_nu[nu] + a_mu[mu]) % Ns,
                    (t - a_nu[nu] + a_mu[mu]) % Nt,
                    nu,
                ]
            )
    return Xmu


@njit(
    complex128[:, :](complex128[:, :, :, :, :, :, :], int64, int64, int64, int64, int64)
)
def ActionDerivative(U, x, y, z, t, mu):
    """See [1] eq.(19)"""
    Wdagger = c0 * staple(x, y, z, t, mu, U) + c1 * Xrect(U, x, y, z, t, mu)
    Wdagger = Wdagger.conj().T

    Omegamu = U[x, y, z, t, mu] @ Wdagger  # [1] eq.(2)

    V = (
        -1j
        * (
            0.5 * (Omegamu - Omegamu.conj().T)
            - 1 / (2 * su3) * np.trace(Omegamu - Omegamu.conj().T)
        )
        * U[x, y, z, t, mu]
    )

    return V


@njit()
def reunitarize(U):
    Q, R = np.linalg.qr(U)
    phase = np.diag(np.exp(1j * np.angle(np.diag(R))))
    U = np.dot(Q, phase)


@njit((complex128[:, :, :, :, :, :, :], float64))
def flow(U, dtau):
    Utemp = np.zeros(U.shape, dtype=complex128)
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    # runge kutta 3rd order
                    for mu in range(4):
                        Utemp[x, y, z, t, mu] = ActionDerivative(U, x, y, z, t, mu)

                    for mu in range(4):
                        U[x, y, z, t, mu] = (
                            expMatrix(1, 0.25 * dtau * Utemp[x, y, z, t, mu])
                            @ U[x, y, z, t, mu]
                        )

                        # print(np.linalg.det(U[x, y, z, t, mu]))

                    for mu in range(4):
                        Utemp[x, y, z, t, mu] = (
                            ActionDerivative(U, x, y, z, t, mu) * (8 / 9) * dtau
                            - Utemp[x, y, z, t, mu] * (17 / 36) * dtau
                        )

                    for mu in range(4):
                        U[x, y, z, t, mu] = (
                            expMatrix(1, Utemp[x, y, z, t, mu]) @ U[x, y, z, t, mu]
                        )

                    for mu in range(4):
                        Utemp[x, y, z, t, mu] = (
                            ActionDerivative(U, x, y, z, t, mu) * 0.75 * dtau
                            - Utemp[x, y, z, t, mu]
                        )

                    for mu in range(4):
                        U[x, y, z, t, mu] = (
                            expMatrix(1, Utemp[x, y, z, t, mu]) @ U[x, y, z, t, mu]
                        )

                        # unitarize(U[x, y, z, t, mu])
                        print(np.linalg.det(U[x, y, z, t, mu]))


def mainFlow():

    U = initialize_lattice(1)
    for i in range(10):
        U = HB_updating_links(beta, U)
    Wils = []
    topolo = []
    for _ in range(Nstepflow):
        W = WilsonAction(R, T, U)

        print(W)
        print("flowing:", _)
        flow(U, dtauflow)
        Q = q(U)
        topolo.append(Q)
        Wils.append(W)

    plt.figure()
    plt.scatter(np.arange(Nstepflow), Wils, marker="x")
    plt.show()

    plt.figure()
    plt.scatter(np.arange(Nstepflow), topolo, marker="x")
    plt.show()

    # np.savetxt("topologico.txt", topolo)


if __name__ == "__main__":
    # U = initialize_lattice(1)
    # Utemp = ActionDerivative(U, 0, 0, 0, 0, 0)
    # U1 = expMatrix(1, Utemp)

    # print(np.linalg.det(U1))

    mainFlow()
    # top = np.loadtxt("topologico.txt")
    # plt.figure()
    # plt.hist(top, bins=FreedmanDiaconis(top), histtype="step")
    # plt.show()
