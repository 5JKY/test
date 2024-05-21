import os
import numpy as np
import matplotlib.pyplot as plt

from fokker_planck_solver import FokkerPlanckSolover_ReRe, FokkerPlanckSolover_AbAb, FokkerPlanckSolover_AbRe, FokkerPlanckSolover_ReAb

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

def beta_Up(x):
    # result = 20*x-10  # convex potential
    # result = -20*x+10  # concave potential
    # result = -3*(20.0 - 25.0*x)*np.exp(-12.5*(x - 0.8)**2)-2*(25.0 - 100.0*x)*np.exp(-50.0*(x - 0.25)**2) # Gaussian potential
    result = 0*x
    return result


def beta_Upp(x):
    # result = 20  # convex potential
    # result = -20  # concave potential
    # result = -2*10000.0*(0.25 - x)**2*np.exp(
    #     -50.0*(x - 0.25)**2
    # ) - 3*625.0*(0.8 - x)**2*np.exp(
    #     -12.5*(x - 0.8)**2
    # ) + 200.0*np.exp(
    #     -50.0*(x - 0.25)**2
    # ) + 75.0*np.exp(
    #     -12.5*(x - 0.8)**2)  # Gaussian potential
    result = 0*x
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
u0[:] = initial_condition(x_arr, L/4)

# Potential energy on all Nx spatial nodes
beta_Up_arr = beta_Up(x_arr)
beta_Upp_arr = beta_Upp(x_arr)
D_arr = D*np.ones(x_arr.size)


phi = 1/2
gamma = 1/2

Tf = 50  # Final simulation time for 10 lines plot
tf = 15  # final time for compare different boudary conditions

aa_sol = FokkerPlanckSolover_AbAb(
    ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)
print(aa_sol.plot_solution(Tf=Tf, freq=10))

ra_sol = FokkerPlanckSolover_ReAb(
    ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)
print(ra_sol.plot_solution(Tf=Tf, freq=10))


ar_sol = FokkerPlanckSolover_AbRe(
    ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)
print(ar_sol.plot_solution(Tf=Tf, freq=10))

rr_sol = FokkerPlanckSolover_ReRe(
    ht, hx, x_arr, u0, beta_Up_arr, beta_Upp_arr, D_arr, phi, gamma)
print(rr_sol.plot_solution(Tf=Tf, freq=10))

plt.plot(x_arr, aa_sol.solve(Tf=tf), label=f"t={tf:.2f} AbAb", marker="o", linestyle='-', markevery=20)
plt.plot(x_arr, ra_sol.solve(Tf=tf), label=f"t={tf:.2f} ReAb", marker="s", linestyle='-', markevery=30)
plt.plot(x_arr, ar_sol.solve(Tf=tf), label=f"t={tf:.2f} AbRe", marker="^", linestyle='-', markevery=50)
plt.plot(x_arr, rr_sol.solve(Tf=tf), label=f"t={tf:.2f} ReRe", linestyle='-')

# Plot formatting
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Compare Solution of FPE with different BCs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(folder_path, 'numerical_boundaries.png'))  # Save as PNG file

plt.show()

# longer time, leaking quantities
print(rr_sol.plot_solution(Tf=1000, freq=10))