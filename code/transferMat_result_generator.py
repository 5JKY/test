import os
import numpy as np
import matplotlib.pyplot as plt

from transfer_matrix_reptile import TransferMatrix_ReRe, TransferMatrix_ReAb, TransferMatrix_AbRe, TransferMatrix_AbAb


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))



# Parameters
L = 1.0  # Length of the domain
D = 0.01  # Diffusion coefficient

hx = 0.01  # Spatial grid spacing
Nx = int(L / hx) + 1  # Number of spatial grid points

x_arr = np.linspace(0, L, Nx)
ht = 0.005  # Time step size

# Initial condition (delta function)
def initial_condition(x, x0):
    return np.where(np.abs(x-x0) < hx/2, 1.0/hx, 0.0)

# Initialize solution array
pi0 = np.zeros(Nx)
# Set initial condition (delta function at x = L/4)
x_arr = np.linspace(0, L, Nx)
pi0[:] = initial_condition(x_arr, L/4)

Tf = 50

def beta_U(x):
    # result = 10*x**2-10*x  # convex potential
    # result = -10*x**2+10*x  # concave potential
    result = gaussian(x, A=-2, mu=L/4, sigma=0.1)+gaussian(x, A=-3, mu=0.8, sigma=0.2) # Gaussian potential
    # result = 0*x # diffusion only
    return result

# ## =========first run with Metro=============
# rr_trans = TransferMatrix_ReRe(hx, x_arr, beta_U, 0)
# aa_trans = TransferMatrix_AbAb(hx, x_arr, beta_U, 0)
# ra_trans = TransferMatrix_ReAb(hx, x_arr, beta_U, 0)
# ar_trans = TransferMatrix_AbRe(hx, x_arr, beta_U, 0)

# # Define the folder path
# # folder_path = 'graphs/convex_potential/transferMat_Metro'
# # folder_path = 'graphs/concave_potential/transferMat_Metro'
# # folder_path = 'graphs/Gaussian_potential/transferMat_Metro'
# # folder_path = 'graphs/diffusion_only/transferMat_Metro'



# # ===========second run param: with different criteria============
rr_trans = TransferMatrix_ReRe(hx, x_arr, beta_U, 1)
aa_trans = TransferMatrix_AbAb(hx, x_arr, beta_U, 1)
ra_trans = TransferMatrix_ReAb(hx, x_arr, beta_U, 1)
ar_trans = TransferMatrix_AbRe(hx, x_arr, beta_U, 1)

# Define the folder path
# folder_path = 'graphs/convex_potential/transferMat_Criteria'
# folder_path = 'graphs/concave_potential/transferMat_Criteria'
folder_path = 'graphs/Gaussian_potential/transferMat_Criteria'
# folder_path = 'graphs/diffusion_only/transferMat_Criteria'


plt.plot(x_arr, rr_trans.steady_state, label="RR")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the first LM Eigenvalue(= {rr_trans.eig6_w[0].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_steady_rr.png'))  # Save as PNG file
plt.show() ##########

plt.plot(x_arr, 1.0/(hx*np.sum(rr_trans.eig6_v[:, 1]))*rr_trans.eig6_v[:, 1], label="RR")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the second LM Eigenvalue(= {rr_trans.eig6_w[1].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_eigv2_rr.png'))  # Save as PNG file
plt.show() ##########

plt.plot(x_arr[1:-1], aa_trans.steady_state, label="AA")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the first LM Eigenvalue(= {aa_trans.eig6_w[0].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_steadyno_aa.png'))  # Save as PNG file
plt.show() #######


plt.plot(x_arr[1:-1], 1.0/(hx*np.sum(aa_trans.eig6_v[:, 1]))*aa_trans.eig6_v[:, 1], label="AA")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the second LM Eigenvalue(= {aa_trans.eig6_w[1].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_eigv2_aa.png'))  # Save as PNG file
plt.show() #######


plt.plot(x_arr[:-1], ra_trans.steady_state, label="RA")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the first LM Eigenvalue(= {ra_trans.eig6_w[0].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_steadyno_ra.png'))  # Save as PNG file
plt.show() ########


plt.plot(x_arr[:-1], 1.0/(hx*np.sum(ra_trans.eig6_v[:, 1]))*ra_trans.eig6_v[:, 1], label="RA")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the second LM Eigenvalue(= {ra_trans.eig6_w[1].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_eigv2_ra.png'))  # Save as PNG file
plt.show() #######


plt.plot(x_arr[1:], ar_trans.steady_state, label="AR")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the first LM Eigenvalue(= {ar_trans.eig6_w[0].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_steadyno_ar.png'))  # Save as PNG file
plt.show() ########


plt.plot(x_arr[1:], 1.0/(hx*np.sum(ar_trans.eig6_v[:, 1]))*ar_trans.eig6_v[:, 1], label="AR")
# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title(f'The Eigenvector of the second LM Eigenvalue(= {ar_trans.eig6_w[1].real.round(6)})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save the figure
plt.savefig(os.path.join(folder_path, 'transferMat_eigv2_ar.png'))  # Save as PNG file
plt.show() #######


# manually save 4 times in one corresonpind potential directory

rr_trans.plot_eigenvalues() # os.path.join(folder_path, 'transferMat_eigws_rr.png')
aa_trans.plot_eigenvalues() # os.path.join(folder_path, 'transferMat_eigws_aa.png')
ra_trans.plot_eigenvalues() # os.path.join(folder_path, 'transferMat_eigws_ra.png')
ar_trans.plot_eigenvalues() # os.path.join(folder_path, 'transferMat_eigws_ar.png')

# Again
## manually save 4 times in one corresonpind potential directory
rr_trans.plot_evolution_pi(pi0, Tf, freq=10)  # os.path.join(folder_path, 'transferMat_evolve_rr.png')
aa_trans.plot_evolution_pi(pi0, Tf, freq=10)  # folder_path, 'transferMat_evolve_aa.png'
ra_trans.plot_evolution_pi(pi0, Tf, freq=10) # folder_path, 'transferMat_evolve_ra.png'
ar_trans.plot_evolution_pi(pi0, Tf, freq=10)  # folder_path, 'transferMat_evolve_ar.png'


tf=15
plt.plot(x_arr, rr_trans.evolve_pi(pi0, Tf=tf), label=f"t={tf:.2f} RR", marker="o", linestyle='-', markevery=20)
plt.plot(x_arr[:-1], ra_trans.evolve_pi(pi0, Tf=tf), label=f"t={tf:.2f} RA", marker="s", linestyle='-', markevery=30)
plt.plot(x_arr[1:], ar_trans.evolve_pi(pi0, Tf=tf), label=f"t={tf:.2f} AR", marker="*", linestyle='-', markevery=50)
plt.plot(x_arr[1:-1], aa_trans.evolve_pi(pi0, Tf=tf), label=f"t={tf:.2f} AA")

# Plot formatting
plt.xlabel('x')
plt.ylabel('$ \pi(x,t) $')
plt.title('Compare Matrix Evolved Prob Distribution with different BCs')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
#plt.savefig(os.path.join(folder_path, 'transferMat_boudaries.png'))  # Save as PNG file

plt.show()
