from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t = np.linspace(0, 10, 1001)

def EqOfMovement(y, t, M, m, r, c, g):
    # y[0, 1, 2, 3] = fi, phi, fi', phi'
    # dy[0, 1, 2, 3] = fi', phi', fi'', phi''
    dy = np.zeros_like(y)
    dy[0] = y[2]
    dy[1] = y[3]
    a11 = 1
    a12 = np.cos(y[0])
    a21 = np.cos(y[0])
    a22 = 1+2*(M/m)
    b1 = -2*(c/m)*(1-np.cos((y[0]+y[1])/2))*np.sin((y[0]+y[1])/2)-(g/r)*np.sin(y[0])
    b2 = y[2]**2*np.sin(y[0]) - 2*(c/m)*(1-np.cos((y[0]+y[1])/2))*np.sin((y[0]+y[1])/2)

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)
    return dy


r = 1
M = 1
m = 2
c = 40
g = 9.81

fi0 = np.pi/3
phi0 = 0
dfi0 = 0
dphi0 = 0

y0 = [fi0, phi0, dfi0, dphi0]

Y = odeint(EqOfMovement, y0, t, (M, m, r, c, g))

fi = Y[:, 0]  
phi = Y[:, 1]
dfi = Y[:, 2]
dphi = Y[:, 3]

ddfi = np.array([EqOfMovement(yi, ti, M, m, r, c, g)[2] for yi, ti in zip(Y, t)])
ddphi = np.array([EqOfMovement(yi, ti, M, m, r, c, g)[3] for yi, ti in zip(Y, t)])

N = m * (g*np.cos(fi) + r*(dfi**2-ddphi**2*np.sin(fi))) + 2*r*c*np.cos((fi+phi)/2)*(1-np.cos((fi+phi)/2))

fig_for_graphs = plt.figure(figsize=[13, 7])  # построим их графики
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, fi, color='blue')
ax_for_graphs.set_title("fi(t)")
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, phi, color='red')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0, 10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,3)
ax_for_graphs.plot(t,dfi,color='green')
ax_for_graphs.set_title("fi\'(t)")
ax_for_graphs.set(xlim=[0,10])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2,2,4)
ax_for_graphs.plot(t,dphi,color='black')
ax_for_graphs.set_title('phi\'(t)')
ax_for_graphs.set(xlim=[0,10])
ax_for_graphs.grid(True)

fig_for_N = plt.figure(figsize=[13,7])  # построим их графики
ax_for_N = fig_for_N.add_subplot(1, 1, 1)
ax_for_N.plot(t, N, color='blue')
ax_for_N.set_title("N(t)")
ax_for_N.set(xlim=[0, 10])
ax_for_N.grid(True)
fig = plt.figure(figsize=[6.5, 6.5])
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim=[-20, 20], ylim=[-20, 20])

def Rot2D(X,Y,Phi):
    RotX = X*np.cos(Phi) - Y*np.sin(Phi)
    RotY = X*np.sin(Phi) + Y*np.cos(Phi)
    return RotX, RotY


steps = 1000
t = np.linspace(0, 10, steps)

phi = np.sin(t)
psi = np.cos(1.2 * t)


R1 = 5; R2 = 4
OsX1 = -15; OsX2 = 15
OsY1 = OsY2 = -R1
x_C = 0; y_C = 0

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
b = 1/(n-2)
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
