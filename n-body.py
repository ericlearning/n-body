import numpy as np
import matplotlib.pyplot as plt
from euler import euler
from improved import improved_euler
from rungekutta import rungekutta
from scipy.integrate import odeint

## n-body problem setup

# gravitational constant
G = 1

# masses
m = [10, 2, 75]

# coordinates
r = [(2.5, 1), (-1, 2), (0, -4)]

# initial velocity
v = [(5, -2), (0, 3), (1, 1)]

# number of objects
N = len(m)

m, r, v = map(np.array, (m, r, v))

def pairwise_dist(r):
    # r: (n, 2)
    x, y = r[:, :1], r[:, 1:]
    pairwise_dist_x = (x - x.T)**2
    pairwise_dist_y = (y - y.T)**2
    dist = pairwise_dist_x + pairwise_dist_y
    dist **= 0.5
    return dist

def pairwise_disp(r):
    # r: (n, 2)
    x, y = r[:, :1], r[:, 1:]
    pairwise_disp_x = (x - x.T)
    pairwise_disp_y = (y - y.T)
    return pairwise_disp_x, pairwise_disp_y

def pairwise_mass(m):
    # m: (n)
    pairwise_mass = m[:, None] @ m[None, :]
    return pairwise_mass

def component(G, r, m):
    dist = pairwise_dist(r)
    disp_x, disp_y = pairwise_disp(r)
    mass = pairwise_mass(m)
    comp_x = -G * mass * disp_x / (dist ** 3 + 1e-7)
    comp_y = -G * mass * disp_y / (dist ** 3 + 1e-7)
    return comp_x, comp_y

def acceleration(G, r, m):
    comp_x, comp_y = component(G, r, m)
    return comp_x.sum(1), comp_y.sum(1)

def n_body(y, _, G, m):
    # y: (12) -> (2, 3, 2)
    y = y.reshape(2, N, 2)
    r, v = y

    # (3, 2)
    dr_dt = v
    dv_dt = np.stack(acceleration(G, r, m), -1)

    # derivative: (2, 3, 2)
    derivative = np.stack([dr_dt, dv_dt], 0)
    return derivative.flatten()

## simulate

# step size 0.1
h = 0.01

# evaluate t between t_min and t_max
t_min, t_max = 0, 10
total_num = int((t_max-t_min)/h)
t = np.linspace(t_min, t_max, num=total_num)

init = np.stack([r, v], 0)

# calculate position and velocity across time
out = odeint(n_body, init.flatten(), t, args=(G, m))
out = out.reshape(-1, 2, N, 2)
r, v = out[:, 0], out[:, 1]
r = r.transpose(1, 0, 2)

# plotting the points
for (x, y) in r[0]:
    plt.plot(x, y, '-o', color='red')
for (x, y) in r[1]:
    plt.plot(x, y, '-o', color='green')
for (x, y) in r[2]:
    plt.plot(x, y, '-o', color='blue')
plt.show()