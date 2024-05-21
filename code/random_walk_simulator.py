import numpy as np
import matplotlib.pyplot as plt

class RandomWalkers_Metro:
    def __init__(self, ht, hx, L, num_walkers, init_position_arr, beta_U):
        self.ht = ht
        self.hx = hx
        self.L = L
        self.num_particles = num_walkers
        self.init_position_arr = init_position_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U

        self.D0 = hx**2/(2*ht)


    def simulate_AbAb(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        num_steps = int(Tf/self.ht)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, criteria: min[1, exp(-beta*delta_U)]
            accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of values within the range: (Absorb, Absorb)
            indices = np.where(
                (position_arr > 0.0+step_size/4) & (position_arr < self.L-step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        unique_values, counts = np.unique(position_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations


    def simulate_ReAb(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        num_steps = int(Tf/self.ht)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, criteria: min[1, exp(-beta*delta_U)]
            accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of walkers outside reflecting boundary
            lower_reflect_id = np.where(position_arr < 0.0-step_size/4)
            position_arr[lower_reflect_id] += step_size
            # Use np.where() to find indices of values within the range: [reflect, absorb) 
            indices = np.where(
                (position_arr > 0.0-step_size/4) & (position_arr < self.L-step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        # Round values to a specific precision (e.g., 4 decimal places as hx=0.01)
        rounded_arr = np.round(position_arr, decimals=4)
        unique_values, counts = np.unique(rounded_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations


    def simulate_AbRe(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        num_steps = int(Tf/self.ht)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, criteria: min[1, exp(-beta*delta_U)]
            accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of walkers outside reflecting boundary
            upper_reflect_id = np.where(position_arr > self.L+step_size/4)
            position_arr[upper_reflect_id] -= step_size
            # Use np.where() to find indices of values within the range: (absorb, reflect]
            indices = np.where(
                (position_arr > 0.0+step_size/4) & (position_arr < self.L+step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        # Round values to a specific precision (e.g., 4 decimal places as hx=0.01)
        rounded_arr = np.round(position_arr, decimals=4)
        unique_values, counts = np.unique(rounded_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations


    def simulate_ReRe(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        num_steps = int(Tf/self.ht)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, criteria: min[1, exp(-beta*delta_U)]
            accept_move = np.random.rand(position_arr.size) < np.exp(-energy_difference_arr)
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of walkers outside reflecting boundary
            lower_reflect_id = np.where(position_arr < 0.0-step_size/4)
            position_arr[lower_reflect_id] += step_size
            upper_reflect_id = np.where(position_arr > self.L+step_size/4)
            position_arr[upper_reflect_id] -= step_size
            # Use np.where() to find indices of values within the range: [reflect, reflect]
            indices = np.where(
                (position_arr > 0.0-step_size/4) & (position_arr < self.L+step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        # Round values to a specific precision (e.g., 4 decimal places as hx=0.01)
        rounded_arr = np.round(position_arr, decimals=4)
        unique_values, counts = np.unique(rounded_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations



class RandomWalkers_Criteria:
    def __init__(self, ht, hx, L, num_walkers, init_position_arr, beta_U):
        self.ht = ht
        self.hx = hx
        self.L = L
        self.num_particles = num_walkers
        self.init_position_arr = init_position_arr
        # beta_U is a function, can be called by beta_U(arr)
        self.beta_U = beta_U
        # half of the original Metropolis criteria's defination of diffusion coefficient
        # But we will use delt_t = ht/2, slow the time to keep D0 match 
        self.D0 = hx**2/(4*ht)

    def simulate_AbAb(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        delt_t = 0.5*self.ht
        num_steps = int(Tf/delt_t)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, analytical criteria
            accept_move = np.random.rand(
                position_arr.size) < 1.0/(np.exp(energy_difference_arr)+1.0)
            
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of values within the range: (Absorb, Absorb)
            indices = np.where(
                (position_arr > 0.0+step_size/4) & (position_arr < self.L-step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        unique_values, counts = np.unique(position_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations


    def simulate_ReAb(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        delt_t = 0.5*self.ht
        num_steps = int(Tf/delt_t)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, analytical criteria
            accept_move = np.random.rand(
                position_arr.size) < 1.0/(np.exp(energy_difference_arr)+1.0)
            
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of walkers outside reflecting boundary
            lower_reflect_id = np.where(position_arr < 0.0-step_size/4)
            position_arr[lower_reflect_id] += step_size
            # Use np.where() to find indices of values within the range: [reflect, absorb) 
            indices = np.where(
                (position_arr > 0.0-step_size/4) & (position_arr < self.L-step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        # Round values to a specific precision (e.g., 4 decimal places as hx=0.01)
        rounded_arr = np.round(position_arr, decimals=4)
        unique_values, counts = np.unique(rounded_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations


    def simulate_AbRe(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        delt_t = 0.5*self.ht
        num_steps = int(Tf/delt_t)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, analytical criteria
            accept_move = np.random.rand(
                position_arr.size) < 1.0/(np.exp(energy_difference_arr)+1.0)
            
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of walkers outside reflecting boundary
            upper_reflect_id = np.where(position_arr > self.L+step_size/4)
            position_arr[upper_reflect_id] -= step_size
            # Use np.where() to find indices of values within the range: (absorb, reflect]
            indices = np.where(
                (position_arr > 0.0+step_size/4) & (position_arr < self.L+step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        # Round values to a specific precision (e.g., 4 decimal places as hx=0.01)
        rounded_arr = np.round(position_arr, decimals=4)
        unique_values, counts = np.unique(rounded_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations


    def simulate_ReRe(self, Tf):
        position_arr = np.copy(self.init_position_arr)
        delt_t = 0.5*self.ht
        num_steps = int(Tf/delt_t)
        step_size = self.hx

        # Generate random steps
        for i in np.arange(num_steps):
            # Metropolis algorithm
            trial_position_arr = position_arr + np.random.choice(
                [-step_size, step_size], size=position_arr.size
                )
            # Calculate beta * delta_U(potential energy difference: U_new-U_old)
            energy_difference_arr = self.beta_U(trial_position_arr) - self.beta_U(position_arr)
            # condition for accepting the move, analytical criteria
            accept_move = np.random.rand(
                position_arr.size) < 1.0/(np.exp(energy_difference_arr)+1.0)
            
            position_arr[accept_move] = trial_position_arr[accept_move]
            # Use np.where() to find indices of walkers outside reflecting boundary
            lower_reflect_id = np.where(position_arr < 0.0-step_size/4)
            position_arr[lower_reflect_id] += step_size
            upper_reflect_id = np.where(position_arr > self.L+step_size/4)
            position_arr[upper_reflect_id] -= step_size
            # Use np.where() to find indices of values within the range: [reflect, reflect]
            indices = np.where(
                (position_arr > 0.0-step_size/4) & (position_arr < self.L+step_size/4)
                )
            # Select values within the range using the indices
            position_arr = position_arr[indices]

        # Round values to a specific precision (e.g., 4 decimal places as hx=0.01)
        rounded_arr = np.round(position_arr, decimals=4)
        unique_values, counts = np.unique(rounded_arr, return_counts=True)
        # concentrations in every spacial step
        concentrations = counts/(self.hx*self.num_particles)
        return unique_values, concentrations
