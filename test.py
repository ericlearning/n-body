import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# step size 0.1
h = 0.1

# evaluate t between t_min and t_max
t_min, t_max = 0, 10
t = np.linspace(t_min, t_max, num=int((t_max-t_min)/h))

# initial condition
# (t = 0, y = 7)
init = 7

# ODE (Ordinary Differential Equation)
# (dy/dt = 1-x)
# (y = Ce^(-t)+1)
def function(y, t):
    return 1 - y

def solution(t, c=6):
    return c * np.exp(-t) + 1

# solve the ODE
sol = odeint(function, init, t)
plt.plot(t, sol[:, 0], color='red')
plt.plot(t, solution(t), color='blue')
plt.show()