import numpy as np
import matplotlib.pyplot as plt
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
gt1 = f1_sol(t, *f1_args_ex, init)

sol2 = odeint(f2, init, t, args=f2_args_ex)
gt2 = f2_sol(t, *f2_args_ex, init)

fig, ax = plt.subplots(2)
ax[0].plot(t, sol1[:, 0], color='red')
ax[0].plot(t, gt1, color='blue')
ax[1].plot(t, sol2[:, 0], color='red')
ax[1].plot(t, gt2, color='blue')
plt.show()