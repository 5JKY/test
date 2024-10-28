# Revolution Version with Markov Chain Extended
import numpy as np
import matplotlib.pyplot as plt


def simulate_AbInAb_regular(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    """
    b1: position of the lower absorbing boundary 
    b2: position of the upper absorbing boundary
    """
    position_arr = init_point * np.ones(num_particles, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=8)
    init_id = np.where(n_arr == init_point)
    # count the number of particles fall on initial position, to obtain Pst(x0)
    count_n = np.zeros(n_arr.size)
    count_n[init_id] = num_particles
    # 2d array of first passage time: each raw is for one experiment, each column is for one accounted position
    ti_n = np.zeros((num_particles, n_arr.size))
    abort_ti_n = np.array([])

    # iteration numbers, can be converted to time
    left_iter_count_arr = np.zeros(num_particles)
    right_iter_count_arr = np.zeros(num_particles)
    flag_escape_arr = np.zeros(num_particles)
    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice(
            [-step_size, step_size], size=position_arr.size
            )
        # Calculate beta * delta_U(potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            # Find indices where the obtained position fall on the n_arr
            idx_n = np.where(n_arr == position_arr[i])
            if idx_n[0].size == 1:
                count_n[idx_n] += 1
                if position_arr[i] < init_point-step_size/2:
                    left_iter_count_arr[i] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = left_iter_count_arr[i]
                elif position_arr[i] > init_point+step_size/2:
                    right_iter_count_arr[i] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = right_iter_count_arr[i]
                else:
                    left_iter_count_arr[i] += 1
                    right_iter_count_arr[i] += 1
            
        # Use mask to find indices of walkers (reach the left absorbing boundar) & (need to be recycled)
        left_return_mask = (position_arr < b1+step_size/4) & (flag_escape_arr<1)
        flag_escape_arr[left_return_mask] -= 1
        # counting the contribution of left recycling step 
        position_arr[left_return_mask] = init_point
        left_iter_count_arr[left_return_mask] += 1
        right_iter_count_arr[left_return_mask] += 1
        count_n[init_id] += np.sum(left_return_mask)
        # Each simulation is aborted when the walker escape from left after visiting right absoring boundary
        indices = np.where(position_arr < b1+step_size/4)
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)
        flag_escape_arr = np.delete(flag_escape_arr, indices)
        
        # Use mask to find indices of walkers (reach the right absorbing boundar) & (need to be recycled)
        right_return_mask = (position_arr > b2-step_size/4) & (flag_escape_arr>-1)
        flag_escape_arr[right_return_mask] += 1
        # counting the contribution of right recycling step 
        position_arr[right_return_mask] = init_point
        left_iter_count_arr[right_return_mask] += 1
        right_iter_count_arr[right_return_mask] += 1
        count_n[init_id] += np.sum(right_return_mask)
        # Each simulation is aborted when the walker escape from right after visiting left absoring boundary
        indices = np.where(position_arr > b2-step_size/4)
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)
        flag_escape_arr = np.delete(flag_escape_arr, indices)

    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n, abort_ti_n

