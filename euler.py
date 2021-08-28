def euler(f, y0, ts, args=()):
    ys = [y0]
    n = len(ts)
    for tn_prev, tn in zip(ts[:-1], ts[1:]):
        h = tn - tn_prev
        yn = ys[-1]
        yn = yn + h * f(yn, tn_prev, *args)
        ys.append(yn)
    return ys