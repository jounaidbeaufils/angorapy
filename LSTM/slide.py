import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D

N = 12
pts = np.random.rand(N, 3)
startpts = np.random.rand(N, 3)
color = np.random.randint(N/2, N*2, 3)
color = np.repeat(color, 4)
# line = matplotlib.path.Path(, codes=2)
lines = np.linspace(pts, startpts, 5)

def curved_line(point1, point2):
    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 100)
    y = a*np.cosh(x) + b
    z = np.linspace(point1[2], point2[2], 100)
    return (x,y, z)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
           marker='x', s=20, c=color)
for i in range(N):
    # ax.plot3D(xlin[:, i, 0], ylin[:, i, 1], lines[:, i, 2])
    lines = curved_line(pts[i, :], startpts[i, :])
    ax.plot3D(lines[0], lines[1], lines[2])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()
