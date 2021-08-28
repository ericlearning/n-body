import numpy as np
from scipy.integrate import odeint

## n-body problem setup

# masses
m = [10, 2, 75]

# coordinates
x = [(2.5, 1), (-1, 1), (0, -4)]

# initial velocity
v = [(5, -2), (0, 3), (1, 1)]

