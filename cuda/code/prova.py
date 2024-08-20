import numpy as np
import matplotlib.pyplot as plt

latticedim = 5120

def plot2d():
    
    plot = True
    
    
    lattice = np.loadtxt("final_lattice.txt", dtype=np.int32)
    print("Lattice shape: ", lattice.shape)
    
    if (plot):
        plt.imshow(lattice)
        plt.title('Initial Lattice Configuration')
        plt.colorbar()
        plt.show()


def plot3d():
    import numpy as np
    from mayavi import mlab

    # Carica il reticolo 3D dal file
    
    nx, ny, nz = latticedim, latticedim, latticedim # Specifica le dimensioni del reticolo
    lattice = np.loadtxt("final.txt", dtype=np.int32).reshape(nx, ny, nz)

    # Visualizzazione volumetrica con Mayavi
    src = mlab.pipeline.scalar_field(lattice)
    mlab.pipeline.iso_surface(src, contours=[0], opacity=0.3)
    mlab.pipeline.iso_surface(src, contours=[lattice.max()-0.1], color=(1, 1, 1))
    mlab.show()
    
def plot3dSlides():

    # Carica il reticolo 3D dal file
    nx, ny, nz = latticedim, latticedim, latticedim  # Specifica le dimensioni del reticolo
    lattice = np.loadtxt("final.txt", dtype=np.int32).reshape(nx, ny, nz)

    # Visualizza le slice lungo la dimensione z
    for z in range(5):
        plt.figure()
        plt.imshow(lattice[:, :, z], cmap='viridis')
        plt.title(f'Slice at z = {z}')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    # plot3dSlides()
    plot2d()