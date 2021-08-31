import numpy as np
import matplotlib.pyplot as plt
from euler import euler
from improved import improved_euler
from rungekutta import rungekutta
from scipy.integrate import odeint

## n-body problem setup

# gravitational constant
G = 1

# number of objects
n_obj = 15

# masses
m = [10, 2, 50]
m = np.random.uniform(10, 80, (n_obj,))

# coordinates
r = [(2.5, 1), (-1, 2), (0, -4)]
r = np.random.uniform(-5, 5, (n_obj, 2))

# initial velocity
v = [(5, -2), (0, 3), (1, 1)]
v = np.random.uniform(-6, 6, (n_obj, 2))

# number of objects
N = len(m)

m, r, v = map(np.array, (m, r, v))

def remove_diag(x):
    n = x.shape[0]
    return x[~np.eye(n, dtype='bool')].reshape(n, n-1)

def pairwise_dist(r):
    # r: (n, 2)
    x, y = r[:, :1], r[:, 1:]
    pairwise_dist_x = (x - x.T)**2
    pairwise_dist_y = (y - y.T)**2
    dist = pairwise_dist_x + pairwise_dist_y
    dist = remove_diag(dist)
    dist **= 0.5
    return dist

def pairwise_disp(r):
    # r: (n, 2)
    x, y = r[:, :1], r[:, 1:]
    pairwise_disp_x = (x - x.T)
    pairwise_disp_y = (y - y.T)
    pairwise_disp_x = remove_diag(pairwise_disp_x)
    pairwise_disp_y = remove_diag(pairwise_disp_y)
    return pairwise_disp_x, pairwise_disp_y

def pairwise_mass(m):
    # m: (n)
    m = m[None].repeat(m.shape[0], 0)
    m = remove_diag(m)
    return m

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
t_min, t_max = 0, 5
total_num = int((t_max-t_min)/h)
t = np.linspace(t_min, t_max, num=total_num)

init = np.stack([r, v], 0)

# calculate position and velocity across time
out = rungekutta(n_body, init.flatten(), t, args=(G, m))
out = out.reshape(-1, 2, N, 2)
r, v = out[:, 0], out[:, 1]
r = r.transpose(1, 0, 2)

# plotting the points
for cur_r in r:
    cur_color = np.random.rand(3)
    for (x, y) in cur_r:
        if -50 <= x <= 50 and -50 <= y <= 50:
            plt.plot(x, y, '-o', color=cur_color)
plt.show()