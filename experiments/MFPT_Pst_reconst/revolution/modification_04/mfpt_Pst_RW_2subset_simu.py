# The enhanced version of AbInAb_regular simulation
import numpy as np
import matplotlib.pyplot as plt
import time


def simulate_AbInAb_regular(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    """
    b1: position of the lower absorbing boundary 
    b2: position of the upper absorbing boundary
    """
    # active_walkers_num =int(num_particles/1)
    active_walkers_num = 1
    position_arr = init_point * np.ones(active_walkers_num, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=8)
    # slice require integer
    init_id = np.where(n_arr == init_point)[0][0]
    n1_arr = n_arr[ :init_id+1]
    n2_arr = n_arr[init_id: ]
    # count the number of particles fall on the left region amd right region seperately
    count_n1 = np.zeros((active_walkers_num, n1_arr.size))
    count_n2 = np.zeros((active_walkers_num, n2_arr.size))
    # count the number of particles fall on initial position, to obtain Pst(x0)
    count_n1[:, -1] += 1
    count_n2[:, 0] += 1
    abort_count_n1 = np.array([])
    abort_count_n2 = np.array([])
    # 2d array of first passage time: each raw is for one experiment, each column is for one accounted position
    fpt_n1 = np.zeros((active_walkers_num, n1_arr.size))
    fpt_n2 = np.zeros((active_walkers_num, n2_arr.size))
    abort_fpt_n1 = np.array([])
    abort_fpt_n2 = np.array([])
    # iteration numbers, can be converted to time
    left_iter_count_arr = np.zeros(active_walkers_num)
    right_iter_count_arr = np.zeros(active_walkers_num)
    # record the number of valid aborts
    abort1_num = 0
    abort2_num = 0
    # Keep generating random steps until at least num_particles of the walkers escape from [a, b)
    # while (abort1_num < num_particles) or (abort2_num < num_particles):

    # Set time limit (in seconds)
    time_limit = 60*10
    start_time = time.time()
    while True:
        if time.time() - start_time > time_limit:
            # abort1_num += active_walkers_num
            # abort_count_n1 = np.append(abort_count_n1, count_n1)
            # abort_fpt_n1 = np.append(abort_fpt_n1, fpt_n1)

            # abort2_num += active_walkers_num
            # abort_count_n2 = np.append(abort_count_n2, count_n2)
            # abort_fpt_n2 = np.append(abort_fpt_n2, fpt_n2)
            break
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice(
            [-step_size, step_size], size=position_arr.size
            )
        # Calculate beta * delta_U (potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            if position_arr[i] < init_point-step_size/2:
                # Find indices where the obtained position fall on the n1_arr
                idx_n1 = np.where(n1_arr == position_arr[i])
                if idx_n1[0].size == 1:
                    left_iter_count_arr[i] += 1
                    count_n1[i, idx_n1] += 1
                    if fpt_n1[i, idx_n1] == 0:
                        fpt_n1[i, idx_n1] = left_iter_count_arr[i]
                    if position_arr[i] < b1+step_size/4:
                        abort1_num += 1
                        print(abort1_num, 'left')
                        abort_count_n1 = np.append(abort_count_n1, count_n1[i])
                        abort_fpt_n1 = np.append(abort_fpt_n1, fpt_n1[i])
                        count_n1[i] = np.zeros(n1_arr.size)
                        fpt_n1[i] = np.zeros(n1_arr.size)
                        left_iter_count_arr[i] = 0
                        position_arr[i] = init_point
                        left_iter_count_arr[i] += 1
                        right_iter_count_arr[i] += 1
                        count_n1[i, -1] += 1
                        count_n2[i, 0] += 1
            elif position_arr[i] > init_point+step_size/2:
                # Find indices where the obtained position fall on the n2_arr
                idx_n2 = np.where(n2_arr == position_arr[i])
                if idx_n2[0].size == 1:
                    right_iter_count_arr[i] += 1
                    count_n2[i, idx_n2] += 1
                    if fpt_n2[i, idx_n2] == 0:
                        fpt_n2[i, idx_n2] = right_iter_count_arr[i]
                    if position_arr[i] > b2-step_size/4:
                        abort2_num += 1
                        print(abort2_num, 'right')
                        abort_count_n2 = np.append(abort_count_n2, count_n2[i])
                        abort_fpt_n2 = np.append(abort_fpt_n2, fpt_n2[i])
                        count_n2[i] = np.zeros(n2_arr.size)
                        fpt_n2[i] = np.zeros(n2_arr.size)
                        right_iter_count_arr[i] = 0
                        position_arr[i] = init_point
                        left_iter_count_arr[i] += 1
                        right_iter_count_arr[i] += 1
                        count_n1[i, -1] += 1
                        count_n2[i, 0] += 1
            else:
                left_iter_count_arr[i] += 1
                right_iter_count_arr[i] += 1
                count_n1[i, -1] += 1
                count_n2[i, 0] += 1
    
    abort_count_n1 = abort_count_n1.reshape((abort1_num, n1_arr.size))
    abort_fpt_n1 = abort_fpt_n1.reshape((abort1_num, n1_arr.size))
    abort_count_n2 = abort_count_n2.reshape((abort2_num, n2_arr.size))
    abort_fpt_n2 = abort_fpt_n2.reshape((abort2_num, n2_arr.size))

    return abort_count_n1, abort_count_n2, abort_fpt_n1, abort_fpt_n2

