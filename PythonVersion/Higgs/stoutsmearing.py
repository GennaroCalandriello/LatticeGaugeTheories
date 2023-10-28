import numpy as np
from numba import njit

"""references:
1. "Stout smearing for openQCD", Jonas Rylund Glesaaen, March 2018
2. "SU(3) matrix functions", Martin Luscher """

su3 = 3


@njit()
def MatrixNormalization(M):

    """normalization"""

    n = 0
    Mnorm = (0.5) ** n * M
    norm = np.linalg.norm(Mnorm)

    while norm > 1:
        Mnorm = (0.5) ** n * M
        norm = np.linalg.norm(Mnorm)
        n += 1
    return Mnorm, n


@njit()
def projectionOperator(V, Ulink):

    """return anti-hermitean traceless matrix, V is the staple, Ulink is the link variable:
    Ulink = U[t, x, y, z, mu]... equation (4) ref. 1"""

    O = V @ Ulink.conj().T
    X = 1 / 2 * (O - O.conj().T) - 1 / 6 * np.trace(O - O.conj().T)

    return X


@njit()
def stoutsmearing(X, p):
    """compute Usmeared=exp(X)*Ulink, p is the coefficient vector, it depends only on tx, dx"""
    Id = np.identity(su3) + 0j
    Y, n = MatrixNormalization(X)
    pnew, dy, ty = coefficients(p, Y)
    # expY=p[0]*np.identity(su3)+p[1]*Y+p[2]*Y@Y
    expX = pnew[0] * Id + p[1] * Y + p[2] * Y @ Y

    # rescaling coefficients:
    dx = 2 ** (3 * n) * dy
    return expX


@njit()
def exponentiation():
    """Compute the exponentiation of the matrix smeared"""
    return 1


@njit()
def coefficients(p, Y):
    """recurrence relation to recalculate the coefficient of the Cayley-Hamilton algorithm for the exponentiation of a matrix"""
    ##come si determinano i primi coefficienti? vedi 2.
    pnew = [0, 0, 0]
    dy = np.linalg.det(Y)
    Ysquare = Y @ Y
    ty = -(1 / 2) * np.trace(Ysquare)

    pnew[0] = p[0] ** 2 - 2 * 1j * dy * p[1] * p[2]
    pnew[1] = 2 * p[0] * p[1] - 1j * dy * p[2] ** 2 - 2 * ty * p[1] * [2]
    pnew[2] = 2 * p[0] * p[2] + p[1] ** 2 - ty * p[2] ** 2

    return pnew, dy, ty

