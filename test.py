
import numpy as np
from scipy.integrate import odeint

# step size 0.1
h = 0.1

# evaluate ys between y_min and y_max
y_min, y_max = -10, 10
ys = np.linspace(-10, 10, num=int((y_max-y_min)/h))

# initial condition
# (t = 0, y = 7)
init = (0, 7)

# ODE (Ordinary Differential Equation)
# (dy/dt = 1-x)
# (y = Ce^(-t)+1)
def test_f(y, t):
    return 1 - y
