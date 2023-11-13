import numpy as np
import matplotlib.pyplot as plt

wilson = np.loadtxt("func/Wilson11.txt")
topological = np.loadtxt("func/TopologicalCharge.txt")
Nstep = len(wilson)

plt.figure(figsize=(10, 10))
plt.scatter(np.arange(Nstep), 1- wilson, label="Wilson", color="blue", marker="+")
plt.title("Wilson, Cooling")
plt.xlabel("Nstep")
plt.ylabel(r"$\langle W \rangle$")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(
    np.arange(Nstep), topological, label="Topological Charge", color="red", marker="+"
)
plt.title("Topological Charge, Cooling")
plt.xlabel("Nstep")
plt.ylabel("Q")
plt.grid(True)
plt.show()
