import numpy as np
import matplotlib.pyplot as plt


# Collect and use all the MFPT both on the left and the right to the initial point-original version
# Need to concern the conditional probability
def simulate_AbInAb(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    """
    b1: position of the lower absorbing boundary
    b2: position of the upper absorbing boundary
    """
    position_arr = init_point * np.ones(num_particles, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=5)
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
    left_iter_count_arr = np.zeros(num_particles)
    right_iter_count_arr = np.zeros(num_particles)
    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
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
        position_arr = np.round(position_arr, decimals=5)
        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            if position_arr[i] < init_point - step_size / 2:
                left_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = left_iter_count_arr[i]
            elif position_arr[i] > init_point + step_size / 2:
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = right_iter_count_arr[i]
            else:
                left_iter_count_arr[i] += 1
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1

        # Use np.where() to find indices of values exceed the two absorbing boundaries
        indices = np.where(
            (position_arr < b1 + step_size / 4) | (position_arr > b2 - step_size / 4)
        )
        # Each simulation is aborted once the walker exceed absorbing boundary
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)

    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n, abort_ti_n


# Discard MFPT on one side which particle exscape from another side
def simulate_AbInAb_edit(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    position_arr = init_point * np.ones(num_particles, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=5)

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
    left_iter_count_arr = np.zeros(num_particles)
    right_iter_count_arr = np.zeros(num_particles)
    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
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
        position_arr = np.round(position_arr, decimals=5)

        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            if position_arr[i] < init_point - step_size / 2:
                left_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = left_iter_count_arr[i]
            elif position_arr[i] > init_point + step_size / 2:
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = right_iter_count_arr[i]
            else:
                left_iter_count_arr[i] += 1
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1

        # Use np.where() to find indices of values exceed the left absorbing boundaries
        indices_left = np.where(position_arr < b1 + step_size / 4)
        if indices_left[0].size > 0:
            ti_n[np.ix_(indices_left[0], n_arr > init_point)] = np.nan
            # Each simulation is aborted once the walker exceed absorbing boundary
            abort_ti_n = np.append(abort_ti_n, ti_n[indices_left[0]])
            ti_n = np.delete(ti_n, indices_left[0], axis=0)
            position_arr = np.delete(position_arr, indices_left[0])
            left_iter_count_arr = np.delete(left_iter_count_arr, indices_left[0])
            right_iter_count_arr = np.delete(right_iter_count_arr, indices_left[0])

        # Use np.where() to find indices of values exceed the right absorbing boundaries
        indices_right = np.where(position_arr > b2 - step_size / 4)
        if indices_right[0].size > 0:
            ti_n[np.ix_(indices_right[0], n_arr < init_point)] = np.nan
            # Each simulation is aborted once the walker exceed absorbing boundary
            abort_ti_n = np.append(abort_ti_n, ti_n[indices_right[0]])
            ti_n = np.delete(ti_n, indices_right[0], axis=0)
            position_arr = np.delete(position_arr, indices_right[0])
            left_iter_count_arr = np.delete(left_iter_count_arr, indices_right[0])
            right_iter_count_arr = np.delete(right_iter_count_arr, indices_right[0])

    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n, abort_ti_n


def simulate_ReInAb(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    """
    b1: position of reflecting boundary (lower boundary)
    b2: position of absorbing boundary (upper boundary)
    """
    position_arr = init_point * np.ones(num_particles, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=5)

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
    left_iter_count_arr = np.zeros(num_particles)
    right_iter_count_arr = np.zeros(num_particles)
    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice(
            [-step_size, step_size], size=position_arr.size
        )
        # Use np.where() to find indices of walkers outside reflecting boundary
        lower_reflect_id = np.where(trial_position_arr < b1 - step_size / 4)
        trial_position_arr[lower_reflect_id] += step_size

        # Calculate beta * delta_U(potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]

        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=5)
        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            if position_arr[i] < init_point - step_size / 2:
                left_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = left_iter_count_arr[i]
            elif position_arr[i] > init_point + step_size / 2:
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = right_iter_count_arr[i]
            else:
                left_iter_count_arr[i] += 1
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1

        # Use np.where() to find indices of values exceed the two absorbing boundaries
        indices = np.where(position_arr > b2 - step_size / 4)
        # Each simulation is aborted once the walker exceed absorbing boundary
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)

    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n, abort_ti_n


def simulate_AbInRe(init_point, num_particles, beta_U, n_arr, b1, b2, hx):
    """
    b1: position of absorbing boundary (lower boundary)
    b2: position of reflecting boundary (upper boundary)
    """
    position_arr = init_point * np.ones(num_particles, dtype=float)
    step_size = hx
    n_arr = np.round(n_arr, decimals=5)
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
    left_iter_count_arr = np.zeros(num_particles)
    right_iter_count_arr = np.zeros(num_particles)
    # Keep generating random steps until all the walkers escape from [a, b)
    while position_arr.size > 0:
        # Metropolis algorithm
        trial_position_arr = position_arr + np.random.choice(
            [-step_size, step_size], size=position_arr.size
        )
        # Use np.where() to find indices of walkers outside reflecting boundary
        upper_reflect_id = np.where(trial_position_arr > b2 + step_size / 4)
        trial_position_arr[upper_reflect_id] -= step_size

        # Calculate beta * delta_U(potential energy difference: U_new-U_old)
        energy_difference_arr = beta_U(trial_position_arr) - beta_U(position_arr)
        # Accepting condition for the move, criteria: min[1, exp(-beta*delta_U)]
        accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
        position_arr[accept_move] = trial_position_arr[accept_move]

        # Round values to a specific precision for better collection and comparison
        position_arr = np.round(position_arr, decimals=5)
        # Collect Pst and MFPT
        for i in np.arange(position_arr.size):
            if position_arr[i] < init_point - step_size / 2:
                left_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = left_iter_count_arr[i]
            elif position_arr[i] > init_point + step_size / 2:
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1
                    if ti_n[i, idx_n] < 1:
                        ti_n[i, idx_n] = right_iter_count_arr[i]
            else:
                left_iter_count_arr[i] += 1
                right_iter_count_arr[i] += 1
                # Find indices where the obtained position fall on the n_arr
                idx_n = np.where(n_arr == position_arr[i])
                if idx_n[0].size == 1:
                    count_n[idx_n] += 1

        # Use np.where() to find indices of values exceed the absorbing boundary
        indices = np.where(position_arr < b1 + step_size / 4)
        # Each simulation is aborted once the walker exceed absorbing boundary
        abort_ti_n = np.append(abort_ti_n, ti_n[indices])
        ti_n = np.delete(ti_n, indices, axis=0)
        position_arr = np.delete(position_arr, indices)
        left_iter_count_arr = np.delete(left_iter_count_arr, indices)
        right_iter_count_arr = np.delete(right_iter_count_arr, indices)

    abort_ti_n = abort_ti_n.reshape((num_particles, n_arr.size))
    return count_n, abort_ti_n
