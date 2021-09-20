"""
A quick and dirty script that plots the curve of the bound
state mass in 2D QED, with respect to the fermion mass, all
in units of e/sqrt(pi).
The numerical integration method is shown (using the module
solve_bndstate) as well as Hamiltonian Truncation.
"""

import truncation
import solve_bndstate

from tqdm import trange
import numpy as np
from matplotlib import pyplot as plt

m = np.linspace(0,2,100)
print("compute using numerical integration. This might take\
    some time...")
M_num = [ solve_bndstate.solve_psi_M(m[i], n=20)[2] for i in trange(0,len(m))]
M_num = np.array(M_num)
print("Computing using Hamiltonian truncation...")
M_trunc = { del_max: truncation.bound_state_mass(2, del_max
    ) for del_max in (3,7,12) }
print("Plotting...")
plt.plot(m, M_num, '--', label="Numerical Integration")
for del_max in M_trunc:
    plt.plot(*M_trunc[del_max],
        label=f"Truncation, $\Delta_M={del_max}$")
plt.legend()
plt.title("Bound State Mass in terms of the Fermion Mass")
plt.xlabel('m')
plt.ylabel('M')
plt.show()
