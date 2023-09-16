import numpy as np
from numba import njit, float64, int64, complex128, boolean

from algebra import *
from randomSU2 import *
import parameters as par

import warnings
from numba.core.errors import NumbaPerformanceWarning

""" Some references:
1.  A MODIFIED HEAT BATH METHOD SUITABLE FOR MONTE CARLO SIMULATIONS
ON VECTOR AND PARALLEL PROCESSORS, K. FREDENHAGEN and M. MARCU"""


# Disable the performance warning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

su3 = par.su3
su2 = par.su2
epsilon = par.epsilon
Ns = par.Ns
Nt = par.Nt
pool_size = par.pool_size


sx = np.array(((0, 1), (1, 0)), dtype=np.complex128)
sy = np.array(((0, -1j), (1j, 0)), dtype=np.complex128)
sz = np.array(((1, 0), (0, -1)), dtype=np.complex128)

####-------------------Initialize configuration------------------------------
def another_initializeLattice():
    """Initialize the lattice with random SU(3) matrices using U =exp(iQ)"""

    import scipy.linalg as la

    U = np.zeros((Ns, Ns, Ns, Nt, 4, 3, 3), dtype=np.complex128)

    def random_su3_matrix():
        omega = np.random.normal(0, 1 / np.sqrt(2), 8)
        # omega = np.random.uniform(0, 1 / np.sqrt(2), size=8)
        hermitian_matrix = np.zeros((3, 3), dtype=np.complex128)

        for i in range(1, 9):
            hermitian_matrix += omega[i - 1] * Tgen(i)
        return la.expm(1j * hermitian_matrix)

    # Populate U
    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        U[x, y, z, t, mu] = random_su3_matrix()
    return U


def initialize_lattice(start):

    """Name says"""

    U = np.zeros((Ns, Ns, Ns, Nt, 4, su3, su3), dtype=np.complex128)
    su3_pool = SU3_pool_generator(pool_size=pool_size)

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        if start == 0:
                            U[x, y, z, t, mu] = np.identity(su3)
                        if start == 1:
                            U[x, y, z, t, mu] = su3_pool[
                                np.random.randint(0, pool_size)
                            ]
    return U


# --------------------------------------------------------------------------------------------------------


def SU3_pool_generator(pool_size):

    """Generate a pool of SU(3) matrices starting from the subgroup SU(2) and projecting on SU(3)"""

    # following pag 83 Gattringer
    su2_pool = SU2_pool_generator(pool_size * 3, epsilon=epsilon)
    su3_pool = np.zeros((pool_size, su3, su3), dtype=np.complex128)

    for i in range(round(pool_size / 2)):  # half pool RST and half RST. conj.T
        r = su2_pool[i]
        s = su2_pool[i + round(pool_size / 2)]
        t = su2_pool[i + 2 * round(pool_size / 2)]

        R = np.array(((r[0, 0], r[0, 1], 0), (r[1, 0], r[1, 1], 0), (0, 0, 1)))
        S = np.array(((s[0, 0], 0, s[0, 1]), (0, 1, 0), (s[1, 0], 0, s[1, 1])))
        T = np.array(((1, 0, 0), (0, t[0, 0], t[0, 1]), (0, t[1, 0], t[1, 1])))

        su3_pool[i] = R @ S @ T
        su3_pool[i + round(pool_size / 2)] = (su3_pool[i].conj().T).copy()

    return su3_pool


@njit()
def SU2SingleMatrix():

    # SU2Matrix = np.array(((0 + 0j, 0 + 0j), (0 + 0j, 0 + 0j)))
    # SU2Matrix = np.empty((2, 2)) + 0j

    r0 = np.random.uniform(-0.5, 0.5)
    x0 = np.sign(r0) * np.sqrt(1 - epsilon**2)

    r = np.random.random((3)) - 0.5
    x = epsilon * r / np.linalg.norm(r)

    SU2Matrix = x0 * np.identity(2) + 1j * x[0] * sx + 1j * x[1] * sy + 1j * x[2] * sz

    return SU2Matrix


@njit()
def checkSU3(O):

    det_O = np.linalg.det(O).real
    det_O = round(det_O)

    if det_O == 1:
        I_alpha = (np.identity(su3) + 0j) * (1)
        Otilde = np.dot(O, I_alpha)

    if det_O == -1:
        I_alpha = (np.identity(su3) + 0j) * (-1)
        Otilde = np.dot(O, I_alpha)

    return Otilde


@njit()
def checkSU2(O):
    det_O = np.linalg.det(O).real
    det_O = round(det_O)

    if det_O == 1:
        I_alpha = (np.identity(su2) + 0j) * (1)
        Otilde = np.dot(O, I_alpha)

    if det_O == -1:
        I_alpha = (np.identity(su2) + 0j) * (-1)
        Otilde = np.dot(O, I_alpha)

    return Otilde


@njit()
def SU3SingleMatrix():

    rr = SU2SingleMatrix()
    ss = SU2SingleMatrix()
    tt = SU2SingleMatrix()

    R = np.array(((rr[0, 0], rr[0, 1], 0), (rr[1, 0], rr[1, 1], 0), (0, 0, 1)))
    S = np.array(((ss[0, 0], 0, ss[0, 1]), (0, 1, 0), (ss[1, 0], 0, ss[1, 1])))
    T = np.array(((1, 0, 0), (0, tt[0, 0], tt[0, 1]), (0, tt[1, 0], tt[1, 1])))

    if np.random.randint(0, 2) == 0:
        SU3Matrix = R @ S @ T
    else:
        SU3Matrix = (R @ S @ T).conj().T

    return SU3Matrix


@njit(
    complex128[:, :](
        int64, int64, int64, int64, int64, complex128[:, :, :, :, :, :, :]
    ),
    fastmath=True,
)
def staple(x, y, z, t, mu, U):

    """Calculate the contribution (interaction) of the 6 links sorrounding the link that we want to update"""
    # njit() works well!

    staple_start = np.array(
        (
            (0 + 0j, 0 + 0j, 0 + 0j),
            (0 + 0j, 0 + 0j, 0 + 0j),
            (0 + 0j, 0 + 0j, 0 + 0j),
        )
    )
    # staple_start = np.empty((3, 3)) + 0j

    a_mu = [0, 0, 0, 0]
    a_mu[mu] = 1

    for nu in range(4):

        if (
            nu == 0 or nu != mu
        ):  # |||||||||||||||WARNING CONTROL EXPERIMENTAL TEST|||||||
            # nu = 0
            # while nu < mu:
            a_nu = [0, 0, 0, 0]
            a_nu[nu] = 1

            # product of 6 matrices
            staple_start += (
                U[
                    (x + a_mu[0]) % Ns,
                    (y + a_mu[1]) % Ns,
                    (z + a_mu[2]) % Ns,
                    (t + a_mu[3]) % Nt,
                    nu,
                ]
                @ U[
                    (x + a_nu[0]) % Ns,
                    (y + a_nu[1]) % Ns,
                    (z + a_nu[2]) % Ns,
                    (t + a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[x, y, z, t, nu].conj().T
            )

            staple_start += (
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
            # nu += 1

    return staple_start


@njit(complex128[:, :](int64), fastmath=True)
def Tgen(a):
    """Generator of su(3) algebra"""
    # njt() works well!
    if a == 1:
        T = np.array((([0, 1, 0]), ([1, 0, 0]), ([0, 0, 0])), dtype=complex128)
    elif a == 2:
        T = np.array((([0, -1j, 0]), ([1j, 0, 0]), ([0, 0, 0])), dtype=complex128)
    elif a == 3:
        T = np.array((([1, 0, 0]), ([0, -1, 0]), ([0, 0, 0])), dtype=complex128)
    elif a == 4:
        T = np.array((([0, 0, 1]), ([0, 0, 0]), ([1, 0, 0])), dtype=complex128)
    elif a == 5:
        T = np.array((([0, 0, -1j]), ([0, 0, 0]), ([1j, 0, 0])), dtype=complex128)
    elif a == 6:
        T = np.array((([0, 0, 0]), ([0, 0, 1]), ([0, 1, 0])), dtype=complex128)
    elif a == 7:
        T = np.array((([0, 0, 0]), ([0, 0, -1j]), ([0, 1j, 0])), dtype=complex128)
    elif a == 8:
        T = np.array(
            (([1, 0, 0]), ([0, 1, 0]), ([0, 0, -2])), dtype=complex128
        ) / np.sqrt(3)
    return T / 2


@njit(complex128[:, :, :, :, :, :, :](), fastmath=True)
def initialMomenta():
    """Initializes momenta with a gaussian distribution
    ref. [4]"""
    # njit() works well! velocissimo
    factor_c = 1  # (Ns**3 * Nt) #decreasing this => decreases the energy fluctiations from configurations
    P = np.zeros((Ns, Ns, Ns, Nt, 4, 3, 3), dtype=np.complex128)

    for x in range(Ns):
        for y in range(Ns):
            for z in range(Ns):
                for t in range(Nt):
                    for mu in range(4):
                        rand = np.random.normal(loc=0, scale=1 / np.sqrt(2), size=8)
                        for a in range(1, 9):
                            P[x, y, z, t, mu] += factor_c * rand[a - 1] * Tgen(a)

    # it's traceless by construction

    return P


@njit()
def staple_calculus(x, y, z, t, mu, U):

    """Calculate the contribution (interaction) of the three links sorrounding the link that we want to update"""
    # staple_start = np.zeros((su3, su3), complex)
    staple_start = np.array(
        (
            (0 + 0j, 0 + 0j, 0 + 0j),
            (0 + 0j, 0 + 0j, 0 + 0j),
            (0 + 0j, 0 + 0j, 0 + 0j),
        )
    )
    # staple_start = np.empty((3, 3)) + 0j

    a_mu = [0, 0, 0, 0]
    a_mu[mu] = 1

    for nu in range(4):

        if mu != nu:
            # nu = 0
            # while nu < mu:
            a_nu = [0, 0, 0, 0]
            a_nu[nu] = 1

            # product of 6 matrices
            staple_start += (
                U[
                    (x + a_mu[0]) % Ns,
                    (y + a_mu[1]) % Ns,
                    (z + a_mu[2]) % Ns,
                    (t + a_mu[3]) % Nt,
                    nu,
                ]
                @ U[
                    (x + a_nu[0]) % Ns,
                    (y + a_nu[1]) % Ns,
                    (z + a_nu[2]) % Ns,
                    (t + a_nu[3]) % Nt,
                    mu,
                ]
                .conj()
                .T
                @ U[x, y, z, t, nu].conj().T
            )

            staple_start += (
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
            # nu += 1
        else:
            continue

    return staple_start


@njit()
def initialize_staple(staple_start):

    """Initialize staple; it's fundamental before each calculus on each link variable"""

    for i in range(su3):
        for j in range(su3):
            staple_start[i, j] = 0 + 0j

    return staple_start


@njit()
def reflectionSU3(U, reflection):

    """This function reflects the off-diagonal element of the matrix for the Over Relaxation algorithm
    Any of the below reflection operations leads to the definition of a new group element having the
    same energy of the original one. The reflection can be selected randomly"""

    if reflection == 1:
        U[0, 1] = -U[0, 1]
        U[1, 0] = -U[1, 0]
        U[0, 2] = -U[0, 2]
        U[2, 0] = -U[2, 0]
    if reflection == 2:
        U[0, 1] = -U[0, 1]
        U[1, 0] = -U[1, 0]
        U[1, 2] = -U[1, 2]
        U[2, 1] = -U[2, 1]
    if reflection == 3:
        U[0, 2] = -U[0, 2]
        U[2, 0] = -U[2, 0]
        U[1, 2] = -U[1, 2]
        U[2, 1] = -U[2, 1]

    return U


@njit()
def det_calculus(W, manual=False):

    """Calculate determinant manually or through linalg"""

    if manual:
        if len(W[:, 0]) == 3:
            det1 = (
                W[0, 0] * W[1, 1] * W[2, 2]
                + W[0, 1] * W[1, 2] * W[2, 0]
                + W[0, 2] * W[1, 0] * W[2, 1]
                - W[0, 2] * W[1, 1] * W[2, 0]
                - W[0, 1] * W[1, 0] * W[2, 2]
                - W[0, 0] * W[1, 2] * W[2, 1]
            )

        if len(W[:, 0]) == 2:
            det1 = W[0, 0] * W[1, 1] - W[0, 1] * W[1, 0]
    else:
        det1 = np.linalg.det(W)

    return det1


@njit()
def normalize(v):

    sum = 0
    for i in range(len(v)):
        sum += v[i] ** 2

    return v / np.sqrt(sum)


@njit()
def GramSchmidt(A, exe):
    """GS orthogonalization, if exe==False => the function normalizes only"""

    n = len(A)

    if exe:
        A[:, 0] = normalize(A[:, 0])

        for i in range(1, n):
            Ai = A[:, i]
            for j in range(0, i):
                Aj = A[:, j]
                t = Ai.dot(Aj)
                Ai = Ai - t * Aj
            A[:, i] = normalize(Ai)
    else:
        for k in range(n):
            # print(np.linalg.norm(A[:, k]))
            A[:, k] = A[:, k] / np.linalg.norm(A[:, k])

    return A


@njit()
def gram_schmidt(A):

    """Gram-Schmidt decomposition"""

    (n, m) = A.shape

    for i in range(m):

        q = A[:, i]  # i-th column of A

        for j in range(i):
            q = q - np.dot(A[:, j], A[:, i]) * A[:, j]

        # if np.array_equal(q, np.zeros(q.shape)):
        #     raise np.linalg.LinAlgError(
        #         "The column vectors are not linearly independent"
        #     )

        # normalize q
        q = q / np.sqrt(np.dot(q, q))

        # write the vector back in the matrix
        A[:, i] = q

    return A


@njit()
def sample_HB_SU2(su2matrix, beta):

    detStaple = np.linalg.det(su2matrix)
    alpha = np.sqrt(detStaple)

    r1 = 1 - np.random.uniform(0, 1)
    r2 = 1 - np.random.uniform(0, 1)
    r3 = 1 - np.random.uniform(0, 1)

    lambda2 = (
        -1
        / (2 * alpha * beta)
        * (np.log(r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(r3))
    )

    while np.random.uniform(0, 1) ** 2 > (1 - lambda2.real):

        r1 = 1 - np.random.uniform(0, 1)
        r2 = 1 - np.random.uniform(0, 1)
        r3 = 1 - np.random.uniform(0, 1)

        lambda2 = (
            -1
            / (2 * alpha * beta)
            * (np.log(r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(r3))
        )

    x0 = 1 - 2 * lambda2

    a0 = x0

    a1 = np.random.uniform(-1, 1)
    a2 = np.random.uniform(-1, 1)
    a3 = np.random.uniform(-1, 1)

    while np.sqrt((a1**2 + a2**2 + a3**2)) > 1:

        a1 = 1 - np.random.uniform(-1, 1)
        a2 = 1 - np.random.uniform(-1, 1)
        a3 = 1 - np.random.uniform(-1, 1)

    # Ulink = np.array(((a0 + 1j * a3, a2 + 1j * a1), (-a2 + 1j * a1, a0 - 1j * a3)))
    Ulink = a0 * np.identity(su2) + 1j * a1 * sx + 1j * a2 * sy + 1j * a3 * sz

    Ulink = GramSchmidt(Ulink, True)
    Ulink = GramSchmidt(Ulink, False)

    return Ulink


@njit()
def quaternion(vec):

    """produces quaternion from a vector of complex and real numbers"""

    # a11 = vec[0] + vec[3] * 1j
    # a12 = vec[2] + vec[1] * 1j
    # a21 = -vec[2] + vec[1] * 1j
    # a22 = vec[0] - vec[3] * 1j
    a11 = complex(vec[0], vec[3])
    a12 = complex(vec[2], vec[1])
    a21 = complex(-vec[2], vec[1])
    a22 = complex(vec[0], -vec[3])

    quat = np.array(((a11, a12), (a21, a22)))
    ##########maybe should be normalized

    return quat


@njit()
def sampleA(a, beta):

    """choose a0 with P(a0) ~ sqrt(1 - a0^2) * exp(beta * k * a0)"""

    w = np.exp(-2 * beta * a)
    xtrial = np.random.uniform(0, 1) * (1 - w) + w
    a0 = 1 + np.log(xtrial) / (beta * a)

    while np.sqrt(1 - a0**2) < np.random.uniform(0, 1):
        xtrial = np.random.uniform(0, 1) * (1 - w) + w
        a0 = 1 + np.log(xtrial) / (beta * a)

    r = np.sqrt(1 - a0**2)
    a1 = np.random.normal()
    a2 = np.random.normal()
    a3 = np.random.normal()

    # while (a1 ** 2 + a2 ** 2 + a3 ** 2) > 1:
    #     a1 = np.random.normal()
    #     a2 = np.random.normal()
    #     a3 = np.random.normal()

    norm = np.sqrt(a1**2 + a2**2 + a3**2)

    a1 = a1 * r / norm
    a2 = a2 * r / norm
    a3 = a3 * r / norm

    avec = np.array((a0, a1, a2, a3))
    avec = normalize(avec)

    return avec


@njit()
def getA(W):

    """Construct the vector needed for the quaternion"""

    a0 = ((W[0, 0] + W[1, 1])).real / 2
    a1 = ((W[0, 1] + W[1, 0])).imag / 2
    a2 = ((W[0, 1] - W[1, 0])).real / 2
    a3 = ((W[0, 0] - W[1, 1])).imag / 2
    Avector = np.array((a0, a1, a2, a3))  # [a0, a1, a2, a3]

    return Avector


@njit()
def HeatbathReconstructed():

    """From: Lattice Simulations of the SU(2) multi-Higgs fields"""

    return 1


@njit()
def PeriodicBC(U, t, x, y, z, N):

    """Impose Periodic Boundary conditions, to be checked, one moment please!"""

    if t == N - 1:
        U[t, x, y, z] = U[0, x, y, z]
    if x == N - 1:
        U[t, x, y, z] = U[t, 0, y, z]
    if y == N - 1:
        U[t, x, y, z] = U[t, x, 0, z]
    if z == N - 1:
        U[t, x, y, z] = U[t, x, y, 0]

    # U[N-1, x, y, z]=U[0, x, y, z]
    # U[t, N-1, y, z]=U[t, 0, y, z]
    # U[t, x, N-1, z]=U[t, x, 0, z]
    # U[t, x, y, N-1]=U[t, x, y, 0]

    return U[t, x, y, z]


@njit(complex128[:, :](complex128, complex128[:, :]), fastmath=True)
def expMatrix(idtau, P):
    method = 2

    if method == 1:
        M = idtau * P
        evals, evecs = np.linalg.eig(M)
        eD = np.diag(np.exp(evals))
        eM = evecs @ eD @ np.linalg.inv(evecs)
    if method == 2:
        # 3x3 identity matrix
        Id = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=complex128)
        # exponential expansion
        eM = (
            Id
            + idtau * P
            + idtau**2 * P @ P / 2
            + idtau**3 * P @ P @ P / 6
            + idtau**4 * P @ P @ P @ P / 24
            + idtau**5 * P @ P @ P @ P @ P / 120
        )

    return eM


@njit()
def ta(mat):
    """Traceless antihermitian part of a matrix"""
    rows, cols = mat.shape

    aux = np.copy(mat)
    trace = 0 + 0j
    one_by_three = 1 / 3

    for i in range(3):
        for j in range(3):
            mat[i][j] = 0.5 * (aux[i][j] - np.conj(aux[j][i]))

    trace = np.trace(mat)
    trace *= one_by_three

    for i in range(3):
        mat[i][i] -= trace


@njit(boolean(complex128[:, :]), fastmath=True)
def antiHermitianVerify(A):
    if A.any() == (-A.conj().T).any():
        return True
    else:
        return False


# project matrix on SU(3)
@njit(fastmath=True)
def unitarize(M):
    """Non lo sooo l'ho copiata da Bonati"""

    norm = 0
    for i in range(3):
        norm += np.real(M[0][i]) * np.real(M[0][i]) + np.imag(M[0][i]) * np.imag(
            M[0][i]
        )
    norm = 1 / np.sqrt(norm)
    for i in range(3):
        M[0][i] *= norm

    prod = 0 + 0j
    for i in range(3):
        prod += np.conj(M[0][i]) * M[1][i]
    for i in range(3):
        M[1][i] -= prod * M[0][i]
    norm = 0
    for i in range(3):
        norm += np.real(M[1][i]) * np.real(M[1][i]) + np.imag(M[1][i]) * np.imag(
            M[1][i]
        )
    norm = 1 / np.sqrt(norm)
    for i in range(3):
        M[1][i] *= norm
    prod = M[0][1] * M[1][2] - M[0][2] * M[1][1]
    M[2][0] = np.conj(prod)
    prod = M[0][2] * M[1][0] - M[0][0] * M[1][2]
    M[2][1] = np.conj(prod)
    prod = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    M[2][2] = np.conj(prod)


# void Su3::sunitarize(void)
#
#   prod=comp[0][1]*comp[1][2]-comp[0][2]*comp[1][1];
#   comp[2][0]=conj(prod);
#   prod=comp[0][2]*comp[1][0]-comp[0][0]*comp[1][2];
#   comp[2][1]=conj(prod);
#   prod=comp[0][0]*comp[1][1]-comp[0][1]*comp[1][0];
#   comp[2][2]=conj(prod);
#   }

if __name__ == "__main__":
    U = initialize_lattice(1)
    print(U.shape)
