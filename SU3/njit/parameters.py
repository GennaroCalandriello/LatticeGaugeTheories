import numpy as np

su3 = 3
su2 = 2
epsilon = 0.2
Ns = 6
Nt = 4
pool_size = Ns**4
measures = 3
idecorrel = 4

N_conf = 5000
Nstep = 40  # number of leapfrog steps
R = 1
T = 1
thermalization = 10

# hybrid monte carlo-----------------------------------
tau = 1  # trajectory length
dtau = tau / Nstep  # 1 / Nstep  # leapfrog step dtau =0.0375 setting from reference [4]
print("dtau:", dtau)
# -------------------------------------------------------

# gradient flow parameters
Nstepflow = 50
tauflow = 7
dtauflow = tauflow / Nstepflow
print("dtauflow:", dtauflow)
# -------------------------------------------------------

# beta_vec = (np.linspace(3.6, 7.8, 20)).tolist()
beta_vec = [3.4, 5.7, 7.9]
beta = 6.9

bins = [2, 4, 8, 16, 32, 64, 128, 256, 512, 660]
bins = np.array(bins, dtype=np.int32)
