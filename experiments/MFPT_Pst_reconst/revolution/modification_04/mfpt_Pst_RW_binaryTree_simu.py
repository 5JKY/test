# The enhanced version of AbInAb_regular simulation
import numpy as np
import matplotlib.pyplot as plt


def simulate_AbInAb_regular(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    """
    b1: position of the lower absorbing boundary 
    b2: position of the upper absorbing boundary
    """
    active_walkers_num = int(num_particles)
    position_arr = init_point * np.ones(active_walkers_num, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=8)
    # slice require integer
    init_id = np.where(n_arr == init_point)[0][0]
    n1_arr = n_arr[ :init_id+1]
    n2_arr = n_arr[init_id: ]
    # count the number of particles fall on the left region and right region seperately
    count_n1 = np.zeros((active_walkers_num, n1_arr.size))
    count_n2 = np.zeros((active_walkers_num, n2_arr.size))
    # count the number of particles fall on initial position, to obtain Pst(x0)
    count_n1[:, -1] += 1
    count_n2[:, 0] += 1
    # 2d array of first passage time: each raw is for one experiment, each column is for one accounted position
    fpt_n1 = np.zeros((active_walkers_num, n1_arr.size))
    fpt_n2 = np.zeros((active_walkers_num, n2_arr.size))
    # iteration numbers, can be converted to time
    left_iter_count_arr = np.zeros(active_walkers_num)
    right_iter_count_arr = np.zeros(active_walkers_num)
    
    # Keep generating random steps until at least num_particles of the walkers escape from [a, b)
    while True:
        continue_mask = (position_arr > b1+step_size/4) & (position_arr < b2-step_size/4)
        indices = np.where(continue_mask)[0]  # Get indices where condition is true
        # Break the loop if no elements satisfy the mask
        if indices.size == 0:
            break
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Metropolis algorithm
        trial_position_arr = position_arr[indices] + np.random.choice(
            [-step_size, step_size], size=indices.size
            )
        # Calculate beta * delta_U (potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr[indices])
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(indices.size) < np.exp(-energy_difference_arr)
        position_arr[indices[accept_move]] = trial_position_arr[accept_move]
        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=8)
        # Collect Pst and MFPT
        for i in indices:
            if position_arr[i] < init_point-step_size/2:
                # Find indices_arr where the obtained position fall on the n1_arr
                idx_n1 = np.where(n1_arr == position_arr[i])[0]
                if idx_n1.size == 1:
                    left_iter_count_arr[i] += 1
                    count_n1[i, idx_n1] += 1
                    if fpt_n1[i, idx_n1] == 0:
                        fpt_n1[i, idx_n1] = left_iter_count_arr[i]
            elif position_arr[i] > init_point+step_size/2:
                # Find indices_arr where the obtained position fall on the n2_arr
                idx_n2 = np.where(n2_arr == position_arr[i])[0]
                if idx_n2.size == 1:
                    right_iter_count_arr[i] += 1
                    count_n2[i, idx_n2] += 1
                    if fpt_n2[i, idx_n2] == 0:
                        fpt_n2[i, idx_n2] = right_iter_count_arr[i]
            else:
                left_iter_count_arr[i] += 1
                right_iter_count_arr[i] += 1
                count_n1[i, -1] += 1
                count_n2[i, 0] += 1

    return count_n1, count_n2, fpt_n1, fpt_n2, left_iter_count_arr, right_iter_count_arr


def synthesize_trajectories(count_left, count_right, fpt_left, fpt_right, left_iter_stop, right_iter_stop):
    count_n1 = np.copy(count_left)
    count_n2 = np.copy(count_right)
    fpt_n1 = np.copy(fpt_left)
    fpt_n2 = np.copy(fpt_right)
    iter_stop_n1 = np.copy(left_iter_stop)
    iter_stop_n2 = np.copy(right_iter_stop)

    harvest_count_n1 = np.array([])
    harvest_count_n2 = np.array([])
    harvest_fpt_n1 = np.array([])
    harvest_fpt_n2 = np.array([])
    
    # queue data structure: FIFO
    queue_count_n1 = np.array([])
    queue_count_n2 = np.array([])
    queue_fpt_n1 = np.array([])
    queue_fpt_n2 = np.array([])
    queue_shifted_timer1 = np.array([])
    queue_shifted_timer2 = np.array([])
    if iter_stop_n1.size == iter_stop_n2.size:
        num_trajectory = iter_stop_n1.size
    for i in np.arange(num_trajectory):
        if fpt_n1[i, 0] == 0:
            # the i-th left trajectory is unfinished, so the i-th right trajectory is a leaf
            if queue_count_n2.shape[0] > 0:
                # synthesize the histograms in the queue with the histogram of the new leaf on the right
                queue_count_n2 += count_n2[i]
                # collect and dequeue all the synthesized histograms 
                harvest_count_n2 = np.append(harvest_count_n2, queue_count_n2)
                queue_count_n2 = np.array([])
            # harvest the histogram of the new leaf on the right
            harvest_count_n2 = np.append(harvest_count_n2, count_n2[i])
            while queue_fpt_n2.shape[0] > 0:
                # shift the starting time of the FPT of the new leaf on the right 
                # which starts counting FPT after the corresponding time in the timer's queue
                shifted_leaf_fpt = fpt_n2[i] + queue_shifted_timer2[0]
                zero_mask = queue_fpt_n2[0] == 0   # Define a mask for the zero elements in queue_fpt_n2[0]
                # synthesize the FPT in the queue with the FPT of the new leaf on the right 
                queue_fpt_n2[0][zero_mask] = shifted_leaf_fpt[zero_mask]
                # collect and dequeue the head of the queue
                harvest_fpt_n2 = np.append(harvest_fpt_n2, queue_fpt_n2[0])
                queue_fpt_n2 = np.delete(queue_fpt_n2, 0, axis=0)  # Deletes the first row
                queue_shifted_timer2= np.delete(queue_shifted_timer2, 0)  # Deletes the first element
            # harvest the new fpt leaf on the right
            harvest_fpt_n2 = np.append(harvest_fpt_n2, fpt_n2[i])
            
            # enqueue the trajectory of the new branch on the left
            if queue_count_n1.shape[0] > 0:
                # synthesize the histograms in the queue with the new branch on the left
                queue_count_n1 += count_n1[i]
            # enqueue the histogram of the new branch on the left
            queue_count_n1 = np.append(queue_count_n1, count_n1[i])
            # keep the 2D array shape of the queue for further synthesis and dequeue
            queue_count_n1 = queue_count_n1.reshape(-1, count_n1.shape[-1])
            if queue_fpt_n1.shape[0] > 0:
                # update all the FPT in the queue with the new branch on the left
                for ii in np.arange(queue_fpt_n1.shape[0]):
                    # shift the starting time of the FPT of the new branch on the left
                    # which starts counting FPT after the last branch's end time in the timers' queue
                    non_zero_mask = fpt_n1[i] != 0   # Define a mask for the non-zero elements in fpt_n1[i]
                    shifted_branch_fpt = np.copy(fpt_n1[i])  # Make a copy to avoid modifying fpt_n1[i] directly
                    # Add ii-th timer in queue only to the non-zero elements
                    shifted_branch_fpt[non_zero_mask] += queue_shifted_timer1[ii]
                    zero_mask = queue_fpt_n1[ii] == 0 # remember to zero the value of FPT at initial position
                    # synthesize the ii-th FPT in the queue with the FPT of the new branch on the left
                    queue_fpt_n1[ii][zero_mask] = shifted_branch_fpt[zero_mask]
                # shift the end time of all the timers in the queue after synthesizing the queue's trajectories
                queue_shifted_timer1 += iter_stop_n1[i]
            # enqueue the FPT of the new branch on the left
            queue_fpt_n1 = np.append(queue_fpt_n1, fpt_n1[i])
            # keep the 2D array shape of the queue for further synthesis and dequeue
            queue_fpt_n1 = queue_fpt_n1.reshape(-1, fpt_n1.shape[-1])
            # enqueue the end time of the new branch into the timers' queue
            queue_shifted_timer1 = np.append(queue_shifted_timer1, iter_stop_n1[i])
        
        if fpt_n2[i, -1] == 0:
            # the i-th right trajectory is unfinished, so the i-th left trajectory is a leaf
            if queue_count_n1.shape[0] > 0:
                # synthesize the histograms in the queue with the histogram of the new leaf on the left
                queue_count_n1 += count_n1[i]
                # collect and dequeue all the synthesized histograms 
                harvest_count_n1 = np.append(harvest_count_n1, queue_count_n1)
                queue_count_n1 = np.array([])
            # harvest the histogram of the new leaf on the left
            harvest_count_n1 = np.append(harvest_count_n1, count_n1[i])
            while queue_fpt_n1.shape[0] > 0:
                # shift the starting time of the FPT of the new leaf on the left 
                # which starts counting FPT after the corresponding time in the timer's queue
                shifted_leaf_fpt = fpt_n1[i] + queue_shifted_timer1[0]
                zero_mask = queue_fpt_n1[0] == 0  # remember to zero the value of FPT at initial position
                # synthesize the FPT in the queue with the FPT of the new leaf on the left 
                queue_fpt_n1[0][zero_mask] = shifted_leaf_fpt[zero_mask]
                # collect and dequeue the head of the queue
                harvest_fpt_n1 = np.append(harvest_fpt_n1, queue_fpt_n1[0])
                queue_fpt_n1 = np.delete(queue_fpt_n1, 0, axis=0)  # Deletes the first row
                queue_shifted_timer1= np.delete(queue_shifted_timer1, 0)  # Deletes the first element
            # harvest the new fpt leaf on the left
            harvest_fpt_n1 = np.append(harvest_fpt_n1, fpt_n1[i])
            
            # enqueue the trajectory of the new branch on the right
            if queue_count_n2.shape[0] > 0:
                # synthesize the histograms in the queue with the new branch on the right
                queue_count_n2 += count_n2[i]
            # enqueue the histogram of the new branch on the right
            queue_count_n2 = np.append(queue_count_n2, count_n2[i])
            # keep the 2D array shape of the queue for further synthesis and dequeue
            queue_count_n2 = queue_count_n2.reshape(-1, count_n2.shape[-1])
            if queue_fpt_n2.shape[0]> 0:
                # update all the FPT in the queue with the new branch on the right
                for ii in np.arange(queue_fpt_n2.shape[0]):
                    # shift the starting time of the FPT of the new branch on the right
                    # which starts counting FPT after the last branch's end time in the timers' queue
                    non_zero_mask = fpt_n2[i] != 0   # Define a mask for the non-zero elements in fpt_n2[i]
                    shifted_branch_fpt = np.copy(fpt_n2[i])  # Make a copy to avoid modifying fpt_n2[i] directly
                    # Add ii-th timer in queue only to the non-zero elements
                    shifted_branch_fpt[non_zero_mask] += queue_shifted_timer2[ii]
                    zero_mask = queue_fpt_n2[ii] == 0 # remember to zero the value of FPT at initial position
                    # synthesize the ii-th FPT in the queue with the FPT of the new branch on the right
                    queue_fpt_n2[ii][zero_mask] = shifted_branch_fpt[zero_mask]
                # shift the end time of all the timers in the queue after synthesizing the queue's trajectories
                queue_shifted_timer2 += iter_stop_n2[i]
            # enqueue the FPT of the new branch on the right
            queue_fpt_n2 = np.append(queue_fpt_n2, fpt_n2[i])
            # keep the 2D array shape of the queue for further synthesis and dequeue
            queue_fpt_n2 = queue_fpt_n2.reshape(-1, fpt_n2.shape[-1])
            # enqueue the end time of the new branch into the timers' queue
            queue_shifted_timer2 = np.append(queue_shifted_timer2, iter_stop_n2[i])
    
    harvest_count_n1 = harvest_count_n1.reshape(-1, count_n1.shape[-1])
    harvest_count_n2 = harvest_count_n2.reshape(-1, count_n2.shape[-1])
    harvest_fpt_n1 = harvest_fpt_n1.reshape(-1, fpt_n1.shape[-1])
    harvest_fpt_n2 = harvest_fpt_n2.reshape(-1, fpt_n2.shape[-1])
    
    return harvest_count_n1, harvest_count_n2, harvest_fpt_n1, harvest_fpt_n2
                
        
