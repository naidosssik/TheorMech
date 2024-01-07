from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure(figsize = [6.5, 6.5])
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim = [-20, 20], ylim = [-20, 20])

def Rot2D(X, Y, Phi):
    RotX = X * np.cos(Phi) - Y * np.sin(Phi)
    RotY = X * np.sin(Phi) + Y * np.cos(Phi)
    return RotX, RotY


steps = 1000
t = np.linspace(0, 10, steps)

phi = np.sin(t)
psi = np.cos(1.2 * t)


R1 = 5
R2 = 4
OsX1 = -15
OsX2 = 15
OsY1 = OsY2 = -R1
x_C = 0
y_C = 0

move = R1 * psi
betta = np.linspace(0, 6.28, 1000)
X_disk1 = R1 * np.sin(betta) + x_C
Y_disk1 = R1 * np.cos(betta) + y_C
X_disk2 = R2 * np.sin(betta) + x_C
Y_disk2 = R2 * np.cos(betta) + y_C

Ax = R1 * np.sin(psi) + move + x_C
Ay = R1 * np.cos(psi) + y_C

r = (R1 - R2) / 2 + R2
Bx = r * np.sin(phi) + move + x_C
By = r * np.cos(phi) + y_C


n = 15
b = 1 / (n - 2)
sh = 0.4
x_P = np.zeros(n)
y_P = np.zeros(n)
x_P[0] = 0
x_P[n-1] = 1
y_P[0] = 0
y_P[n-1] = 0
for i in range(n-2):
    x_P[i+1] = b*(i+1) - b/2
    y_P[i+1] = sh*(-1)**i


katet_x = Bx[i] - Ax[i]
katet_y = By[i] - Ay[i]
stretch = np.sqrt(katet_x ** 2 + katet_y ** 2)
alpha = np.pi + np.arctan2(katet_y, katet_x)
Rx, Ry = Rot2D(x_P * stretch, y_P, alpha)


Centre = ax.plot(x_C + move[0], y_C, 'white', marker='o', ms=10, mec="c")[0]
Line = ax.plot([OsX1, OsX2], [OsY1, OsY2], 'black')
Circle1 = ax.plot(X_disk1 + move[0], Y_disk1, color="c")[0]
Circle2 = ax.plot(X_disk2 + move[0], Y_disk2, color="c")[0]

Spr = ax.plot(Rx + Bx[0], Ry + By[0], 'red')[0]

A = ax.plot(Ax[0], Ay[0], 'k', marker='o', ms=4)[0]
B = ax.plot(Bx[0], Ay[0], 'k', marker='o', ms=8)[0]


def Animation(i):
    Centre.set_data(x_C + move[i], y_C)
    Circle1.set_data(X_disk1 + move[i], Y_disk1)
    Circle2.set_data(X_disk2 + move[i], Y_disk2)
    A.set_data(Ax[i], Ay[i])
    B.set_data(Bx[i], By[i])

    katet_x = Bx[i] - Ax[i]
    katet_y = By[i] - Ay[i]
    stretch = np.sqrt(katet_x ** 2 + katet_y ** 2)
    alpha = np.pi + np.arctan2(katet_y, katet_x)
    Rx, Ry = Rot2D(x_P * stretch, y_P, alpha)

    Spr.set_data(Rx + Bx[i], Ry + By[i])
    return [Circle1, Circle2, Centre, Spr, A, B]

a = FuncAnimation(fig, Animation, frames=steps, interval=10)
plt.show()
