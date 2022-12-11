import numpy as np
from numba import njit

# construction of sigma matrices
sx = np.array(((0, 1), (1, 0)))
sy = np.array(((0, -1j), (1j, 0)))
sz = np.array(((1, 0), (0, -1)))


# @njit()
def SU2_pool_generator(SU2_pool_size, epsilon):
    """I think this generates a certain number of gauge configurations"""

    SU2_pool = (np.zeros((SU2_pool_size, 2, 2), np.complex))

    for i in range(SU2_pool_size):

        r0 = np.random.uniform(-0.5, 0.5)
        x0 = np.sign(r0) * np.sqrt(1 - epsilon ** 2)

        r = np.random.random((3)) - 0.5
        x = epsilon * r / np.linalg.norm(r)

        SU2_pool[i] = (
            x0 * np.identity(2) + 1j * x[0] * sx + 1j * x[1] * sy + 1j * x[2] * sz
        )

    return SU2_pool


@njit()
def sort_from_pool(pool, pool_size):
    index = int((pool_size - 1) * np.random.random())
    return pool[index]

