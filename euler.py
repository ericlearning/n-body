import numpy as np

def euler(f, y0, ts, args=()):
    ys = [y0]
    n = len(ts)
    for tn_prev, tn in zip(ts[:-1], ts[1:]):
        h = tn - tn_prev
        yn_prev = ys[-1]
        yn = yn_prev + h * f(yn_prev, tn_prev, *args)
        ys.append(yn)
    return np.stack(ys, 0)