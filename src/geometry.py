import numpy as np
import matplotlib.pyplot as plt

U_inf = 1
a = 1/3
omega = np.pi/4
R = 1
res = 5
dr = R/res
r_elem = [i*dr for i in range(res)]
x = {r: list() for r in r_elem}
y = {r: list() for r in r_elem}
z = {r: list() for r in r_elem}
U_w = U_inf*(1-a)
for t in np.linspace(0,50,50):
    for r in r_elem:
        x[r].append(U_w*t)
        y[r].append(r*np.sin(omega*t))
        z[r].append(r*np.cos(omega*t))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
for r in r_elem:
    ax.plot(x[r],y[r],z[r])
plt.show()