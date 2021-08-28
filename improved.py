import numpy as np

def improved_euler(f, y0, ts, args=()):
    ys = [y0]
    n = len(ts)
    for tn_prev, tn in zip(ts[:-1], ts[1:]):
        h = tn - tn_prev
        yn_prev = ys[-1]
        yn_euler = yn_prev + h * f(yn_prev, tn_prev, *args)
        yn = yn_prev + h * (f(yn_prev, tn_prev, *args) + f(yn_euler, tn, *args)) / 2
        ys.append(yn)
    return np.stack(ys, 0)