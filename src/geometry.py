import numpy as np
import matplotlib.pyplot as plt

U = 1
x, y, z = list(), list(), list()
omega = np.pi/4
r = 1
for t in np.linspace(0,100,100):
    x.append(U*t)
    y.append(r*np.sin(omega*t))
    z.append(r*np.cos(omega*t))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.plot(x,y,z)
plt.show()