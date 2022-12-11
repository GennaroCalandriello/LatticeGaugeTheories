import numpy as np
import numpy.linalg as la

# construction of sigma matrices
sx = np.array(((0, 1), (1, 0)))
sy = np.array(((0, -1j), (1j, 0)))
sz = np.array(((1, 0), (0, -1)))


def InnerProd(u, v):

    num = False  # use numpy
    """ construct the inner product"""
    if num:
        u = u.conj().T
        inner = np.inner(u, v)

    else:
        inner = 0
        for i in range(len(u)):
            inner += u[i] * v[i]

    return inner


def CrossProd(u, v):
    return np.cross(u, v)


def three_matrix_prod(m1, m2, m3):
    """'Return the product of 3 matrices"""
    prod = np.dot(np.dot(m1, m2), m3)
    return prod


def unitarize(matrix):
    """Unitarize 3x3 matrices"""
    u = matrix[0, :]
    v = matrix[1, :]
    w = matrix[2, :]  # w=u X v

    u = u / la.norm(u)
    v = v - u * (np.dot(v, u.conj()))
    v = v / la.norm(v)
    w = CrossProd(u.conj(), v.conj())
    w = w / la.norm(w)

    matrix[0, :] = u
    matrix[1, :] = v
    matrix[2, :] = w

    return matrix

if __name__=='__main__':
    matrice = np.array(((1j, 2, 2), (3, 0, 1j), (1j, 0, 9)))
    print(unitarize(matrice))
