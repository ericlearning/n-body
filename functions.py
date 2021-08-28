import numpy as np

# (dy/dt = n-y)
# (y = Ce^(-t)+n)

f1_args_ex = (12,)

def f1(y, t, n):
    return n - y

def f1_sol(t, n, init=0):
    C = init - n
    return C * np.exp(-t) + n

# (dy/dt = t^2)
# (y = 1/3 t^3)

f2_args_ex = ()

def f2(y, t):
    return t ** 2

def f2_sol(t, init=0):
    C = init
    return (1/3) * (t**3) + C
