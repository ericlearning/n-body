import numpy as np
from scipy.integrate import odeint

## n-body problem setup

# masses
m = [10, 2, 75]

# coordinates
r = [(2.5, 1), (-1, 1), (0, -4)]

# initial velocity
v = [(5, -2), (0, 3), (1, 1)]

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