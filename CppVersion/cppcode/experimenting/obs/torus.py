import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define angles for parametric plot
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
c, a, z1 = 1, 0.5, 1  # c is the center radius of the torus, a is the tube radius

# Torus parametric equations
X = (c + a * np.cos(theta)) * np.cos(phi)
Y = (c + a * np.cos(theta)) * np.sin(phi)
Z = z1 * np.sin(theta)

# Now creating the closed path on the surface
# We will fix a value of phi to create a loop on the torus surface
t = np.linspace(0, 2 * np.pi, 100)  # Parameter t going from 0 to 2*pi
R, r = 1.0, 0.5
# Define the theta and phi as functions of t for a figure-eight shape
theta1 = R * np.sin(t)
phi1 = t  # Adjust this for the specific shape

# Calculate the coordinates of the path on the torus surface
X_path = (R + r * np.cos(theta1)) * np.cos(phi1)
Y_path = (R + r * np.cos(theta1)) * np.sin(phi1)
Z_path = r * np.sin(theta1)

# Create the figure and axis
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))

# Plot the torus surface
ax.plot_surface(
    X, Y, Z, color="lightskyblue", rstride=5, cstride=5, alpha=0.5, edgecolor="grey"
)

# Plot the closed path on the surface of the torus
# ax.plot(X_path, Y_path, Z_path, color='red', linewidth=2)

# Set labels
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

ax.set_zlim(-2, 2)

# Adjust the view angle
ax.view_init(elev=20.0, azim=20)

# Display the plot
# plt.savefig('high_def_torus.png', dpi=300, bbox_inches='tight', pad_inches=0) #high resolution
plt.show()
