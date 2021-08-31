import time
import arcade
import numpy as np
import matplotlib.pyplot as plt
from rungekutta import rungekutta_step

## n-body problem setup

# gravitational constant
G = 1

# number of objects
n_obj = 50

# masses
m = [10, 2, 50]
m = np.random.uniform(1, 1, (n_obj,))

# coordinates
r = [(2.5, 1), (-1, 2), (0, -4)]
r = np.random.randn(n_obj, 2) * 20

# initial velocity
v = [(5, -2), (0, 3), (1, 1)]
v = np.random.randn(n_obj, 2) * 0.01

# number of objects
N = len(m)

m, r, v = map(np.array, (m, r, v))

def map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

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
    comp_x = -G * mass * disp_x / (dist ** 3 + 1)
    comp_y = -G * mass * disp_y / (dist ** 3 + 1)
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

# current time
cur_t = 0

class NBody(arcade.Window):
    def __init__(self, w, h, title, r, v, cur_t, step_size):
        super().__init__(w, h, title)
        self.w, self.h = w, h
        self.cur_t, self.step_size = cur_t, step_size
        self.state = np.stack([r, v], 0).flatten()
        self.objects = None
    
    def setup(self):
        arcade.set_background_color(arcade.color.WHITE)

    def step(self):
        self.state = rungekutta_step(n_body, self.cur_t+self.step_size,
            self.cur_t, self.state, (G, m))
        self.cur_t += self.step_size
        return self.state.reshape(2, N, 2)[0]

    def on_key_press(self, symbol, _):
        if symbol == arcade.key.I:
            r = np.random.randn(n_obj, 2) * 20
            v = np.random.randn(n_obj, 2) * 0.01
            self.state[:N * 2] = r.flatten()
            self.state[N * 2:] = v.flatten()
    
    def on_draw(self):
        arcade.start_render()
        pos = self.step()
        min_coord, max_coord = -100, 100
        for (x, y) in pos:
            if min_coord < x < max_coord and min_coord < y < max_coord:
                cur_x = map(x, min_coord, max_coord, 0, self.w)
                cur_y = map(y, min_coord, max_coord, 0, self.h)
                arcade.draw_circle_filled(
                    cur_x, cur_y, 5, (255, 0, 0))

WIDTH = 800
HEIGHT = 800

def main():
    window = NBody(
        w=WIDTH,
        h=HEIGHT,
        title='N-Body Simulator',
        r=r,
        v=v,
        cur_t=cur_t,
        step_size=h
    )
    window.set_update_rate(1/1000)
    window.setup()
    arcade.run()

if __name__ == '__main__':
    main()