import numpy as np
import matplotlib.pyplot as plt


def plot_():
    betaArr = [6.2]

    plt.figure(figsize=(10, 10))
    plt.title("Wilson, cooling")
    for b in betaArr:
        W = np.loadtxt(f"W{b}.txt")

        nstep = len(W)

        # plot W
        plt.plot(
            np.arange(nstep),
            1 - W,
            "v",
            label=r"$\beta$" f"={b}",
            markersize=5,
            linestyle="--",
        )
        plt.xlabel("nstep")
        plt.ylabel("W")
        plt.grid(True)
        plt.legend()
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.title("Q, cooling")
    i = 0
    for b in betaArr:
        i += 1
        Q = np.loadtxt(f"Q{b}.txt")
        nstep = len(Q)

        # plot Q
        plt.plot(
            np.arange(nstep),
            Q,
            label=f"conf. {i}",
            marker="v",
            markersize=1.5,
            linestyle="--",
        )
        plt.xlabel("nstep")
        plt.ylabel("Q")
        plt.grid(True)
        plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_()
