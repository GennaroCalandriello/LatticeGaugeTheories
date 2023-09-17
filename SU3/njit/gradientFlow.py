from topological_charge import *
from parameters import *

"""Gradient (Wilson) Flow for SU(3) gauge theory.
References: 
[1] Bonati D'Elia: Comparison of the gradient 
                   flow with cooling in SU(3) pure gauge theory
                   
[2] M. L\"{u}scher: Trivializing maps, the Wilson flow and
                    the HMC algorithm
                    
[3] M. Luscher: Properties and uses of the Wilson flow in lattice QCD
"""


@njit(complex128[:, :, :, :, :, :, :](complex128[:, :, :, :, :, :, :], float64))
def Z_U(U, beta):
    "Udot = Z_t[U]*U"
    # njit test passed

    V = np.zeros((Ns, Ns, Ns, Nt, 4, 3, 3), dtype=complex128)
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        Omegamu = (
                            U[x, y, z, t, mu] @ (staple(x, y, z, t, mu, U)).conj().T
                        )  # [1] eq.(2)

                        V[x, y, z, t, mu] = (
                            -(beta / (2 * su3))
                            * (
                                0.5 * (Omegamu - Omegamu.conj().T)
                                - 1 / (2 * su3) * np.trace(Omegamu - Omegamu.conj().T)
                            )
                        ) * U[
                            x, y, z, t, mu
                        ]  # [1] eq.(3)

    return V


@njit(
    complex128[:, :, :, :, :, :, :](
        complex128[:, :, :, :, :, :, :], complex128[:, :, :, :, :, :, :]
    )
)
def expZ_dot_V(V1, V2):
    """See function Runge Kutta for the definition of V1 and V2"""
    # njit test passed
    exp_Z = np.zeros(V1.shape, dtype=complex128)

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        exp_Z[x, y, z, t, mu] = (
                            expMatrix(1, V1[x, y, z, t, mu]) @ V2[x, y, z, t, mu]
                        )

    return exp_Z


@njit(
    complex128[:, :, :, :, :, :, :](complex128[:, :, :, :, :, :, :], float64, float64),
    fastmath=True,
)
def RungeKutta(U, beta, dt):
    """RungeKutta 3rd order method for solving Udot = Z_t[U]*U.
    This returns U(t+dt) given U(t) and dt. Valid for only one step.
    It must be iterated along the flow time

    ref. of the integration scheme => [3] appendix C"""

    # 1. step
    V0 = U

    # 2. calculate Z_0 = Z(U)
    Z_0 = dt * Z_U(V0, beta)

    # 3.  V1 = exp(1/4 Z(V0))V0
    V1 = expZ_dot_V(0.25 * Z_0, V0)

    # 4. step Z_1 = Z(V1)
    Z_1 = dt * Z_U(V1, beta)

    # 5. V2 = exp(8/9 Z_1 - 17/36 Z_0)V1
    V2 = expZ_dot_V((8 / 9) * Z_1 - (17 / 36) * Z_0, V1)

    # 6. step Z_2 = Z(V2)
    Z_2 = dt * Z_U(V2, beta)

    # 7. V3 = exp(3/4 Z_2 - 8/9 Z_1 + 17/36 Z_0)V2
    V3 = expZ_dot_V((3 / 4) * Z_2 - (8 / 9) * Z_1 + (17 / 36) * Z_0, V2)

    return V3


@njit(fastmath=True)
def UnitarizeConfiguration(C):

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        unitarize(C[x, y, z, t, mu])


def main():
    """ "For each value of the bare coupling we have generated
    O(104) configurations, each one separated from the
    next by 200 Monte Carlo steps, a single step consisting
    of a full lattice update with 1 heatbath ([23, 24]) and
    5 overrelaxation sweeps ([25]). On these configurations,
    we have evaluated the topological charge after smoothing,
    by using both cooling (we have reached a maximum of 50
    cooling steps, with measurements taken after each step)
    and the gradient flow (reaching a maximum flow time
    τ = 10, with measurements performed every ∆τ = 0.2)" [1]"""

    U = initialize_lattice(1)

    # perform Nstep of RungeKutta (see parameters)
    topological_charges = []
    for conf in range(N_conf):
        print("conf:", conf)

        U = HB_updating_links(beta, U)
        U = HB_updating_links(beta, U)

        print("flowing...")
        for _ in range(Nstepflow):
            print("step:", _)
            U = RungeKutta(U, beta, dtauflow)
            UnitarizeConfiguration(U)
        Q = q(U)
        topological_charges.append(Q)
    np.savetxt(f"topological_charges_beta_{beta}.txt", topological_charges)


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    print("tiempo de ejecucion:", round(time.time() - start, 2))
