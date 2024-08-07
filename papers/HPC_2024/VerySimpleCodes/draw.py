import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            row = [float(value) for value in line.split()]
            data.append(row)
    return np.array(data)

def plot_3d(data, title):
    nx, ny = data.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    x, y = np.meshgrid(x, y)
    z = data.T  # Transpose to match the x and y dimensions

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='viridis')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Temperature')

    plt.show()

if __name__ == '__main__':
    initial_data = read_data('initial.dat')
    final_data = read_data('final.dat')

    plot_3d(initial_data, 'Initial Temperature Distribution')
    plot_3d(final_data, 'Final Temperature Distribution')
