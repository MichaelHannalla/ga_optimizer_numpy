import numpy as np
np.random.seed() #Re-seed the random number generator

import matplotlib.pyplot as plt

import random
import math

# Function to convert between binary to decimal in a specified encoding format
def bin2dec(binary_rows, min_val=0, step=1):
    decimal_rows = []
    for row in binary_rows:
        bits_number = len(row)-1
        current_val = min_val
        for idx, val in enumerate(row):
            current_val += (val * (2 ** (bits_number - idx))) * step
        decimal_rows.append(current_val)
    
    decimal_rows = np.asarray(decimal_rows)
    return decimal_rows

# Each variable is encoded in 8 bits
population_bin = np.random.randint(low=0, high=2, size=(10,16))
x_vals_bin = population_bin[:,:8]
y_vals_bin = population_bin[:,8:]

# Conversion to decimal and fitness calculation
x_vals_dec = bin2dec(x_vals_bin)
y_vals_dec = bin2dec(y_vals_bin)
fitness_vals_pop = x_vals_dec**2 + y_vals_dec**2 # We need to minimize the objective/fitness function

# Carrying out iterations and producing new generations
curr_iter = 0
max_iters = 30

# Algorithm hyper-parameters
p_tour = 0.7            # Specifying tournament probability
p_crossover = 0.6       # Specifying crossover probability
p_mut = 0.05            # Specifying mutation probability

# Figure Configuration
X = np.arange(0, 255, 1)
Y = np.arange(0, 255, 1)
X, Y = np.meshgrid(X, Y)
Z = np.sqrt(X**2 + Y**2)
fig = plt.figure()

while curr_iter < max_iters:

    next_gen = np.zeros_like(population_bin)

    for i in range(len(population_bin)): # Produce a new generation with same length of first generation
        # Tournament selection
        idv_1_idx = np.random.randint(low=0, high=len(population_bin))
        idv_2_idx = np.random.randint(low=0, high=len(population_bin))

        fitness_1 = fitness_vals_pop[idv_1_idx]
        fitness_2 = fitness_vals_pop[idv_2_idx]

        r_tour = np.random.rand()

        if fitness_1 >= fitness_2: # Fitness value of individual 1 is higher 
            if r_tour > p_tour:    # Take the worse fitness (higher if minimization, lower if maximization)
                next_gen[i] = population_bin[idv_1_idx, :]
            else:
                next_gen[i] = population_bin[idv_2_idx, :]

        else:
            if r_tour > p_tour:   # Take the worse fitness (higher if minimization, lower if maximization)
                next_gen[i] = population_bin[idv_2_idx, :]
            else:
                next_gen[i] = population_bin[idv_1_idx, :]

    # Crossover between each two pairs of individuals
    crossover_gen = next_gen
    for i in range(0, len(next_gen)-1, 2):
        r_crossover = np.random.rand()
        if r_crossover < p_crossover:
            crossover_point = np.random.randint(low=0, high= next_gen.shape[1])
            a = next_gen[i,:crossover_point].copy()
            b = next_gen[i+1,crossover_point:].copy()
            c = next_gen[i+1,:crossover_point].copy()
            d = next_gen[i,crossover_point:].copy()
            next_gen[i,:] = [*a,*b]
            next_gen[i+1,:] = [*c,*d]
  
    # Mutation
    r_mutation = (np.random.rand(*np.shape(next_gen)) < p_mut).astype(np.int32)
    next_gen = np.logical_xor(next_gen, r_mutation).astype(np.uint8) # Flipping the bits (mutation) where r > p_mut

    # Recalculation of fitness in next generation
    midpoint = next_gen.shape[1]//2
    next_gen_x = next_gen[:, :midpoint]
    next_gen_y = next_gen[:, midpoint:]
    fitness_vals_gen = bin2dec(next_gen_x)**2 + bin2dec(next_gen_y)**2
    
    fitness_vals_all = np.hstack((fitness_vals_pop, fitness_vals_gen))
    population_all = np.vstack((population_bin, next_gen))

    ind = np.argsort(fitness_vals_all)
    ind = ind[:ind.shape[0]//2]

    # Incrementing the iteration and re-writing the population
    population_bin = population_all[ind]
    fitness_vals_pop = fitness_vals_all[ind]

    curr_iter += 1

    # Plotting the algorithm status
    x_dec = bin2dec(population_bin[:,:population_bin.shape[1]//2])
    y_dec = bin2dec(population_bin[:,population_bin.shape[1]//2:])
    plt.contourf(X, Y ,Z)
    plt.plot(x_dec, y_dec, 'ko')
    plt.pause(0.01)
    fig.clf()
    
# Finishing the optimization process and selecting the best solution
sol_idx = np.argmin(fitness_vals_pop)
sol_x_bin = population_bin[sol_idx, :next_gen.shape[1]//2].reshape(1, next_gen.shape[1]//2)
sol_y_bin = population_bin[sol_idx, next_gen.shape[1]//2:].reshape(1, next_gen.shape[1]//2)
sol_x, sol_y = bin2dec(sol_x_bin), bin2dec(sol_y_bin)

print("Finished the optimization process using genetic algorithm optimization")
print("X: " + str(sol_x) + " Y: " + str(sol_y))
print("Current Fitness Value: " + str(sol_x**2 + sol_y**2))

