import numpy as np
import matplotlib.pyplot as plt

wilson = np.loadtxt("func/Wilson11.txt")
topological = np.loadtxt("func/TopologicalCharge.txt")
Nstep = len(wilson)

plt.figure()
plt.scatter(np.arange(Nstep), wilson, label="Wilson", color="blue", marker="+")
plt.show()

plt.figure()
plt.scatter(
    np.arange(Nstep), topological, label="Topological Charge", color="red", marker="+"
)
plt.show()

plt.figure()
plt.hist(topological, bins=10, label="Topological Charge", color="red")
plt.show()
