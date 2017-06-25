from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from math import *
import sys

fig = plt.figure()
ax = fig.gca(projection='3d')

sliceNumber = 1000
epsilon = 2*pi / sliceNumber

# Make data.
X = np.arange(-2*pi, 2*pi, epsilon)
Y = np.arange(-2*pi, 2*pi, epsilon)
X, Y = np.meshgrid(X, Y)

if len(sys.argv)<2:
    raise "please input file name"
else:
    filename = sys.argv[1]

f = open(filename,"r")
Z = np.loadtxt(f, delimiter=", ")
f.close()

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-100.01, 100.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
