import numpy as np
import matplotlib.pyplot as plt

# Orignial slower function for reading and understanding 
def simulate_ReAb(init_position_arr, beta_U, n_arr, a, b, hx, ht):
    position_arr = np.copy(init_position_arr)
    num_particles = position_arr.size
    step_size = hx

    # count the number of particles fall on accounted position, to obtain Pst(x0)
    count_n = np.zeros(n_arr.size)
    for i in np.arange(position_arr.size):
        # Find indices where the array elements are equal to the value
        indices = np.where(n_arr == position_arr[i])
        count_n[indices] += 1
    # 2d array of first passage time: each raw is for one experiment, each column is for one accounted position
    ti_n = np.zeros((num_particles, n_arr.size))
    abort_ti_n = np.array([])

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
                if idx_n[0] != 0 and ti_n[i, idx_n] == 0:
                    ti_n[i, idx_n] = ht*num_iter
       
        # Use np.where() to find indices of values exceed the absorbing boundary
        indices = np.where(position_arr > b-step_size/4)
        # Each simulation is aborted once the walker exceed absorbing boundary
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
    
    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n, abort_ti_n


# Accelerated function for saving time 
def simulate_ReAb_accelerate(init_position_arr, beta_U, n_arr, a, b, hx, ht):
    position_arr = np.copy(init_position_arr)
    num_particles = position_arr.size
    step_size = hx

    # count the number of particles fall on accounted position, to obtain Pst(x0)
    count_n = np.zeros(n_arr.size)
    for i in np.arange(position_arr.size):
        # Find indices where the array elements are equal to the value
        indices = np.where(n_arr == position_arr[i])
        count_n[indices] += 1
    # 2d array of first passage time: each raw is for one experiment, each column is for one accounted position
    ti_n = np.zeros((num_particles, n_arr.size))
    abort_ti_n = np.array([])

    # iteration numbers, can be converted to time
    num_iter = 0
    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
        num_iter += 1
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice([-step_size, step_size], size=position_arr.size)

        # Reflecting boundary condition
        trial_position_arr[trial_position_arr < a - step_size / 4] += step_size

        # Calculate energy differences
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)

        # Accepting condition
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]

        # Round values for comparison
        position_arr = np.round(position_arr, decimals=5)

        # Collect Pst and MFPT
        pos_indices = np.searchsorted(n_arr, position_arr)
        valid_indices = pos_indices < n_arr.size
        valid_positions = position_arr[valid_indices]
        valid_pos_indices = pos_indices[valid_indices]

        # Count and update ti_n
        np.add.at(count_n, valid_pos_indices, 1)
        update_mask = (valid_pos_indices != 0) & (ti_n[np.arange(ti_n.shape[0]), valid_pos_indices] == 0)
        ti_n[np.arange(ti_n.shape[0])[update_mask], valid_pos_indices[update_mask]] = ht * num_iter

        # Absorbing boundary condition
        absorbing_indices = position_arr > b - step_size / 4
        abort_ti_n = np.append(abort_ti_n, ti_n[absorbing_indices])
        ti_n = ti_n[~absorbing_indices]
        position_arr = position_arr[~absorbing_indices]
    
    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n, abort_ti_n

