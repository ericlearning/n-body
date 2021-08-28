import numpy as np

# (dy/dt = n-y)
# (y = Ce^(-t)+n)
def f1(y, t, n):
    return n - y

def f1_sol(t, n, init=0):
    C = init - n
    return C * np.exp(-t) + n

# (dy/dt = t(y^n))
# (y = (((-n+1)/2)t^2+C)^(1/(-n+1)))
def f2(y, t, n):
    return t * y**n

def f2_sol(t, n, init=0):
    C = init ** (-n+1) + (n-1) / 2 * (t**2)
    return ((-n+1) / 2 * (t**2) + C) ** (1 / (-n+1))
