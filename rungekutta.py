import numpy as np

def rungekutta(f, y0, ts, args=()):
    ys = [y0]
    n = len(ts)
    for tn_prev, tn in zip(ts[:-1], ts[1:]):
        h = tn - tn_prev
        yn_prev = ys[-1]

        k1 = h * f(yn_prev, tn_prev, *args)
        k2 = h * f(yn_prev + k1 / 2, tn_prev + h / 2, *args)
        k3 = h * f(yn_prev + k2 / 2, tn_prev + h / 2, *args)
        k4 = h * f(yn_prev + k3, tn, *args)

        yn = yn_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ys.append(yn)
    return np.stack(ys, 0)