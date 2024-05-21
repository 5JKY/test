import os
import numpy as np
import matplotlib.pyplot as plt

from random_walk_simulator import RandomWalkers_Metro, RandomWalkers_Criteria

# Define the folder path
# folder_path = 'graphs/convex_potential'
# folder_path = 'graphs/concave_potential'
# folder_path = 'graphs/Gaussian_potential'
folder_path = 'graphs/diffusion_only'

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def beta_U(x):
    # result = 10*x**2-10*x  # convex potential
    # result = -10*x**2+10*x  # concave potential
    # result = gaussian(x, A=-2, mu=L/4, sigma=0.1)+gaussian(x, A=-3, mu=0.8, sigma=0.2) # Gaussian potential
    result = 0*x 
    return result


# Parameters
D = 0.01  # Diffusion coefficient
L = 1.0  # Length of the domain
hx = 0.01  # Spatial grid spacing
Nx = int(L / hx) + 1  # Number of spatial grid points

ht = 0.005  # Time step size
x_arr = np.linspace(0, L, Nx)

num_particles = 80000 ##### 800000 can be fairly smooth but remeber to change for long time
position_arr = L/4 * np.ones(num_particles, dtype=float)  # Initial position

#################################################################################################
#################################################################################################

walk_ensemble1 = RandomWalkers_Metro(ht, hx, L, num_particles, position_arr, beta_U)

Tf=15
x_aa, y_aa = walk_ensemble1.simulate_AbAb(Tf)
x_ra, y_ra = walk_ensemble1.simulate_ReAb(Tf)
x_ar, y_ar = walk_ensemble1.simulate_AbRe(Tf)
x_rr, y_rr = walk_ensemble1.simulate_ReRe(Tf)

plt.plot(x_aa, y_aa, label=f"t={Tf} Metro walkers (AbAb)")
plt.plot(x_ra, y_ra, label=f"t={Tf} Metro walkers (ReAb)")
plt.plot(x_ar, y_ar, label=f"t={Tf} Metro walkers (AbRe)")
plt.plot(x_rr, y_rr, label=f"t={Tf} Metro walkers (ReRe)")

# Plot formatting
plt.xlabel('x')
plt.ylabel('concentration C(x,t)')
plt.title('random walk with different boundary conditions')
plt.legend()
plt.grid()
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(folder_path, 'simulated_boundaries_metro.png'))  # Save as PNG file

plt.show()

walk_ensemble2 = RandomWalkers_Criteria(ht, hx, L, num_particles, position_arr, beta_U)

Tf=15
x_aa, y_aa = walk_ensemble2.simulate_AbAb(Tf)
x_ra, y_ra = walk_ensemble2.simulate_ReAb(Tf)
x_ar, y_ar = walk_ensemble2.simulate_AbRe(Tf)
x_rr, y_rr = walk_ensemble2.simulate_ReRe(Tf)

plt.plot(x_aa, y_aa, label=f"t={Tf} Criteria walkers (AbAb)")
plt.plot(x_ra, y_ra, label=f"t={Tf} Criteria walkers (ReAb)")
plt.plot(x_ar, y_ar, label=f"t={Tf} Criteria walkers (AbRe)")
plt.plot(x_rr, y_rr, label=f"t={Tf} Criteria walkers (ReRe)")

# Plot formatting
plt.xlabel('x')
plt.ylabel('concentration C(x,t)')
plt.title('random walk with different boundary conditions')
plt.legend()
plt.grid()

plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(folder_path, 'simulated_boundaries_criteria.png'))  # Save as PNG file

plt.show()

#################################################################################################
#################################################################################################


# long time ReRe evolve
# really long time  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# lTf = 1000
# walk_ensemble_metro = RandomWalkers_Metro(ht, hx, L, num_particles, position_arr, beta_U)
# x_rr, y_rr = walk_ensemble_metro.simulate_ReRe(lTf)
# plt.plot(x_rr, y_rr, label=f"t={lTf} Metro walkers (ReRe)")

# # Plot formatting
# plt.xlabel('x')
# plt.ylabel('concentration C(x,t)')
# plt.title('long time Metropolis walk with ReRe bcs')
# plt.legend()
# plt.grid()
# plt.tight_layout()

# # Save the figure
# plt.savefig(os.path.join(folder_path, 'simulated_longTime_metro.png'))  # Save as PNG file

# plt.show()

# walk_ensemble_crit = RandomWalkers_Criteria(ht, hx, L, num_particles, position_arr, beta_U)
# x_rr, y_rr = walk_ensemble_crit.simulate_ReRe(lTf)
# plt.plot(x_rr, y_rr, label=f"t={lTf} Criteria walkers (ReRe)")

# # Plot formatting
# plt.xlabel('x')
# plt.ylabel('concentration C(x,t)')
# plt.title('long time Analytical criteria walk with ReRe bcs')
# plt.legend()
# plt.grid()
# plt.tight_layout()

# # Save the figure
# plt.savefig(os.path.join(folder_path, 'simulated_longTime_criteria.png'))  # Save as PNG file

# plt.show()

# # # really long time  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
