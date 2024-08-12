import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

def simulate_ReAb_count(init_position_arr, beta_U, n_arr, a, b, hx):
    position_arr = np.copy(init_position_arr)
    step_size = hx

    # count the number of particles fall on accounted position, to obtain Pst(x0)
    count_n = np.zeros(n_arr.size)
    for i in np.arange(position_arr.size):
        # Find indices where the array elements are equal to the value
        indices = np.where(n_arr == position_arr[i])
        count_n[indices] += 1

    # iteration numbers, can be converted to time
    num_iter = 0
    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
        num_iter += 1
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice(
            [-step_size, step_size], size=position_arr.size
            )
        # Use np.where() to find indices of walkers outside reflecting boundary
        lower_reflect_id = np.where(trial_position_arr < a-step_size/4)
        trial_position_arr[lower_reflect_id] += step_size
        
        # Calculate beta * delta_U(potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]
        
        # Round values to a specific precision for better collection and comparsion
        position_arr = np.round(position_arr, decimals=5)
        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            # Find indices where the obtained position fall on the n_arr
            idx_n = np.where(position_arr[i] == n_arr)
            if idx_n[0].size == 1:
                count_n[idx_n] += 1
        # Use np.where() to find indices of values exceed the absorbing boundary
        indices = np.where(position_arr > b-step_size/4)
        # Each simulation is aborted once the walker exceed absorbing boundary
        position_arr = np.delete(position_arr, indices)
    return count_n


def simulate_ReAb_fluxIF(init_position_arr, beta_U, n_arr, a, b, hx):
    position_arr = np.copy(init_position_arr)
    num_particles = position_arr.size
    step_size = hx

    # count the number of particles fall on accounted position, to obtain Pst(x0)
    count_n = np.zeros(n_arr.size)
    # 2d array of forward flux: each raw is for one experiment, each column is for one accounted position
    trj_IF = np.zeros((num_particles, n_arr.size))
    abort_trj_IF = np.array([])
    num_trj_FI = np.zeros(num_particles)
    for i in np.arange(position_arr.size):
        # Find indices where the array elements are equal to the value
        indices = np.where(n_arr == position_arr[i])
        count_n[indices] += 1
        trj_IF[i, indices] = 1

    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice(
            [-step_size, step_size], size=position_arr.size
            )
        # Use np.where() to find indices of walkers outside reflecting boundary
        lower_reflect_id = np.where(trial_position_arr < a-step_size/4)
        trial_position_arr[lower_reflect_id] += step_size
        
        # Calculate beta * delta_U(potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]
        
        # Round values to a specific precision for better collection and comparsion
        position_arr = np.round(position_arr, decimals=5)
        init_pos = np.round(init_position_arr, decimals=5)[0]
        # Collect Pst and number of forward and backward flux
        idx_init = np.where(n_arr == init_pos)
        for i in np.arange(position_arr.size):
            # Find indices where the obtained position fall on the n_arr
            idx_n = np.where(n_arr == position_arr[i])
            if idx_n[0].size == 1:
                count_n[idx_n] += 1
                trj_IF[i, idx_n] = 1
                if idx_n[0] == idx_init:
                    num_trj_FI[i] = np.sum(trj_IF[i])
       
        # Use np.where() to find indices of values exceed the absorbing boundary
        indices = np.where(position_arr > b-step_size/4)
        # Each simulation is aborted once the walker exceed absorbing boundary
        abort_trj_IF = np.append(abort_trj_IF, trj_IF[indices])
        trj_IF = np.delete(trj_IF, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
    
    abort_trj_IF = abort_trj_IF.reshape((num_particles, n_arr.size))
    flux_IF = np.sum(abort_trj_IF, axis=0)
    trj_FI = np.zeros((num_particles, n_arr.size))
    for i in np.arange(num_particles):
        trj_FI[i, :int(num_trj_FI[i])] = 1
    flux_FI = np.sum(trj_FI, axis=0)

    return count_n, flux_IF, flux_FI