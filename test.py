import numpy as np
import matplotlib.pyplot as plt
from euler import euler
from improved import improved_euler
from rungekutta import rungekutta
from scipy.integrate import odeint
from functions import f1, f2, f1_sol, f2_sol
from functions import f1_args_ex, f2_args_ex

# step size 0.1
h = 0.1

# evaluate t between t_min and t_max
t_min, t_max = 0, 10
t = np.linspace(t_min, t_max, num=int((t_max-t_min)/h))

# initial condition
# (t = 0, y = 7)
init = 7

# solve the ODE
sol1 = odeint(f1, init, t, args=f1_args_ex)
euler1 = euler(f1, init, t, args=f1_args_ex)
imp1 = improved_euler(f1, init, t, args=f1_args_ex)
rk1 = rungekutta(f1, init, t, args=f1_args_ex)
gt1 = f1_sol(t, *f1_args_ex, init)

sol2 = odeint(f2, init, t, args=f2_args_ex)
euler2 = euler(f2, init, t, args=f2_args_ex)
imp2 = improved_euler(f2, init, t, args=f2_args_ex)
rk2 = rungekutta(f2, init, t, args=f2_args_ex)
gt2 = f2_sol(t, *f2_args_ex, init)

# compare the five methods
fig, ax = plt.subplots(2)
ax[0].plot(t, sol1[:, 0], color='red')
ax[0].plot(t, euler1, color='green')
ax[0].plot(t, imp1, color='yellow')
ax[0].plot(t, rk1, color='orange')
ax[0].plot(t, gt1, color='blue')
ax[1].plot(t, sol2[:, 0], color='red')
ax[1].plot(t, euler2, color='green')
ax[1].plot(t, imp2, color='yellow')
ax[1].plot(t, rk2, color='orange')
ax[1].plot(t, gt2, color='blue')
plt.show()