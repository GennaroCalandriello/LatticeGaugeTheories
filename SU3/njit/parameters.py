import numpy as np

su3 = 3
su2 = 2
epsilon = 0.2
Ns = 9
Nt = 4
pool_size = Ns**4
measures = 3
idecorrel = 4
N_conf = 1000
R = 1
T = 1
thermalization = 200

beta_vec = (np.linspace(4.6, 8.8, 50)).tolist()
# bins = [5 * 2 ** (i - 1) for i in range(1, 10)]
# bins = np.linspace(5, 300, 15)
bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 660]
bins = np.array(bins, dtype=np.int32)
