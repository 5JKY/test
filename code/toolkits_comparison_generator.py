import os
import numpy as np
import matplotlib.pyplot as plt

from fokker_planck_solver import FokkerPlanckSolover_ReRe, FokkerPlanckSolover_AbAb, FokkerPlanckSolover_AbRe, FokkerPlanckSolover_ReAb
from transfer_matrix_reptile import TransferMatrix_ReRe, TransferMatrix_ReAb, TransferMatrix_AbRe, TransferMatrix_AbAb
from random_walk_simulator import RandomWalkers_Metro, RandomWalkers_Criteria

# Define the folder path
# folder_path = 'graphs/convex_potential'
# folder_path = 'graphs/concave_potential'
folder_path = 'graphs/Gaussian_potential'
# folder_path = 'graphs/diffusion_only'


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def beta_U(x):
    # result = 10*x**2-10*x  # convex potential
    # result = -10*x**2+10*x  # concave potential
    result = gaussian(x, A=-2, mu=L/4, sigma=0.1)+gaussian(x, A=-3, mu=0.8, sigma=0.2) # Gaussian potential
    # result = 0*x
    return result

def beta_Up(x):
    # result = 20*x-10  # convex potential
    # result = -20*x+10  # concave potential
    result = -3*(20.0 - 25.0*x)*np.exp(-12.5*(x - 0.8)**2)-2*(25.0 - 100.0*x)*np.exp(-50.0*(x - 0.25)**2) # Gaussian potential
    # result = 0*x
    return result


def beta_Upp(x):
    # result = 20  # convex potential
    # result = -20  # concave potential
    result = -2*10000.0*(0.25 - x)**2*np.exp(
        -50.0*(x - 0.25)**2
    ) - 3*625.0*(0.8 - x)**2*np.exp(
        -12.5*(x - 0.8)**2
    ) + 200.0*np.exp(
        -50.0*(x - 0.25)**2
    ) + 75.0*np.exp(
        -12.5*(x - 0.8)**2)  # Gaussian potential
    # result = 0*x
    return result


# Parameters
D = 0.01  # Diffusion coefficient
L = 1.0  # Length of the domain
hx = 0.01  # Spatial grid spacing
Nx = int(L / hx) + 1  # Number of spatial grid points

ht = 0.005  # Time step size

# Initial condition (delta function)
def initial_condition(x, x0):
    return np.where(np.abs(x-x0) < hx/2, 1.0/hx, 0.0)

# Initialize solution array
u0 = np.zeros(Nx)
# Set initial condition (delta function at x = L/4)
x_arr = np.linspace(0, L, Nx)
u0[:] = initial_condition(x_arr, L/4)  # will also be used in transfer matrix evolve

# Potential energy on all Nx spatial nodes
beta_Up_arr = beta_Up(x_arr)
beta_Upp_arr = beta_Upp(x_arr)
D_arr = D*np.ones(x_arr.size)


tf = 30  # final time for compare !!!!!!!!!!!


phi = 1/2
gamma = 1/2
aa_sol = FokkerPlanckSolover_AbAb(ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)
ra_sol = FokkerPlanckSolover_ReAb(ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)
ar_sol = FokkerPlanckSolover_AbRe(ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)
rr_sol = FokkerPlanckSolover_ReRe(ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)



rr_trans = TransferMatrix_ReRe(hx, x_arr, beta_U, 0)
aa_trans = TransferMatrix_AbAb(hx, x_arr, beta_U, 0)
ra_trans = TransferMatrix_ReAb(hx, x_arr, beta_U, 0)
ar_trans = TransferMatrix_AbRe(hx, x_arr, beta_U, 0)


num_particles = 80000 ##### 800000 can be fairly smooth but remeber to change for long time
position_arr = L/4 * np.ones(num_particles, dtype=float)  # Initial position
walk_ensemble1 = RandomWalkers_Metro(ht, hx, L, num_particles, position_arr, beta_U)
x_aa, y_aa = walk_ensemble1.simulate_AbAb(tf)
x_ra, y_ra = walk_ensemble1.simulate_ReAb(tf)
x_ar, y_ar = walk_ensemble1.simulate_AbRe(tf)
x_rr, y_rr = walk_ensemble1.simulate_ReRe(tf)


##### AbAb
plt.plot(x_arr[1:-1], aa_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_aa, y_aa, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, aa_sol.solve(Tf=tf), label=f"{aa_sol.method_name}", marker="*", linestyle='-', markevery=30)

# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (AbAb Metro)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_metro_{tf:.2f}_aa.png'))  # Save as PNG file
plt.show()


##### ReRe
plt.plot(x_arr, rr_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_rr, y_rr, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, rr_sol.solve(Tf=tf), label=f"{rr_sol.method_name}", marker="*", linestyle='-', markevery=30)

# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (ReRe Metro)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_metro_{tf:.2f}_rr.png'))  # Save as PNG file
plt.show()


########## AbRe
plt.plot(x_arr[1:], ar_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_ar, y_ar, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, ar_sol.solve(Tf=tf), label=f"{ar_sol.method_name}", marker="^", linestyle='-', markevery=30)
# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (AbRe Metro)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_metro_{tf:.2f}_ar.png'))  # Save as PNG file
plt.show()



########## ReAb
plt.plot(x_arr[:-1], ra_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_ra, y_ra, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, ra_sol.solve(Tf=tf), label=f"{ra_sol.method_name}", marker="^", linestyle='-', markevery=30)
# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (ReAb Metro)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_metro_{tf:.2f}_ra.png'))  # Save as PNG file
plt.show()



# ======================================== another criteria===========================================

rr_trans = TransferMatrix_ReRe(hx, x_arr, beta_U, 1)
aa_trans = TransferMatrix_AbAb(hx, x_arr, beta_U, 1)
ra_trans = TransferMatrix_ReAb(hx, x_arr, beta_U, 1)
ar_trans = TransferMatrix_AbRe(hx, x_arr, beta_U, 1)

num_particles = 80000 ##### 800000 can be fairly smooth but remeber to change for long time
position_arr = L/4 * np.ones(num_particles, dtype=float)  # Initial position
walk_ensemble2 = RandomWalkers_Criteria(ht, hx, L, num_particles, position_arr, beta_U)
x_aa, y_aa = walk_ensemble2.simulate_AbAb(tf)
x_ra, y_ra = walk_ensemble2.simulate_ReAb(tf)
x_ar, y_ar = walk_ensemble2.simulate_AbRe(tf)
x_rr, y_rr = walk_ensemble2.simulate_ReRe(tf)


##### AbAb
plt.plot(x_arr[1:-1], aa_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_aa, y_aa, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, aa_sol.solve(Tf=tf), label=f"{aa_sol.method_name}", marker="*", linestyle='-', markevery=30)

# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (AbAb Criteria)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_criteria_{tf:.2f}_aa.png'))  # Save as PNG file
plt.show()


##### ReRe
plt.plot(x_arr, rr_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_rr, y_rr, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, rr_sol.solve(Tf=tf), label=f"{rr_sol.method_name}", marker="*", linestyle='-', markevery=30)

# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (ReRe Criteria)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_criteria_{tf:.2f}_rr.png'))  # Save as PNG file
plt.show()


########## AbRe
plt.plot(x_arr[1:], ar_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_ar, y_ar, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, ar_sol.solve(Tf=tf), label=f"{ar_sol.method_name}", marker="^", linestyle='-', markevery=30)
# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (AbRe Criteria)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_criteria_{tf:.2f}_ar.png'))  # Save as PNG file
plt.show()



########## ReAb
plt.plot(x_arr[:-1], ra_trans.evolve_pi(u0, Tf=tf), label=f"transMat", marker="o", linestyle='-', markevery=20)
plt.plot(x_ra, y_ra, label=f"Metro walk", linestyle='--')
plt.plot(x_arr, ra_sol.solve(Tf=tf), label=f"{ra_sol.method_name}", marker="^", linestyle='-', markevery=30)
# Plot formatting
plt.xlabel('x')
plt.ylabel('P(x,t)')
plt.title(f'Compare 3 Different Approaches at t={tf:.2f} (ReAb Criteria)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, f'approaches_criteria_{tf:.2f}_ra.png'))  # Save as PNG file
plt.show()

