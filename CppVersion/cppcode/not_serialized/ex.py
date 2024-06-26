import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

numproc = 3
proc_list = [0, 1, 2]


def FreedmanDiaconis(data):
    data = np.sort(data)
    n = len(data)
    iqr = data[int(0.75 * n)] - data[int(0.25 * n)]
    h = 2 * iqr * n ** (-1 / 3)
    return int(abs(h))


for i in range(0, 1):
    bw = np.loadtxt(f"breit_wigner_samples_mpi.txt")
    print("lunghezza", len(bw))
    print(FreedmanDiaconis(bw))
    plt.hist(
        bw,
        bins=1000,
        density=True,
        histtype="step",
        label=f"Breit-Wigner {i}",
    )
plt.show()
