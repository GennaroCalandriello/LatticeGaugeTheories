import numpy as np

su3 = 3
su2 = 2
epsilon = 0.2
Ns = 6
Nt = 4
pool_size = Ns**4
measures = 3
idecorrel = 4
N_conf = 50
Nstep = 40  # number of leapfrog steps
R = 1
T = 1
thermalization = 5

# hybrid monte carlo
tau = 1  # trajectory length

dtau = tau / Nstep  # 1 / Nstep  # leapfrog step dtau =0.0375 setting from reference [4]
print("dtau:", dtau)

beta_vec = (np.linspace(3.6, 7.8, 20)).tolist()
# beta_vec = [5.7]
beta = 2.5
# beta_vec = [4.5, 6.7]
# bins = [5 * 2 ** (i - 1) for i in range(1, 10)]
# bins = np.linspace(5, 300, 15)
bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 660]
bins = np.array(bins, dtype=np.int32)
