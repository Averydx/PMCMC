import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from time import perf_counter
from particle_filter import filter
from functools import partial
from scipy.stats import multivariate_normal
from pmcmc import PMCMC
from mm_helper_funcs import *

rng = np.random.default_rng(0)

n=4
pop_test = np.array([500,200])
print("Population per location: ", pop_test)
mov_test = gen_movement(rng,pop_test,min=0.03, max=0.08, Mov=1, Chain=1)

'''Modifying the code to support a particle distribution'''

###Below is setup 
n = 2

num_particles_sim = 1

particles_sim = np.zeros((num_particles_sim,6,60)) #First 4 S, second 4 I, third 4 R
particle_observations_sim = np.zeros((num_particles_sim,4,60))

particles_sim[:,0:2,0] = pop_test
particles_sim[:,2:4,0] = 5
particles_sim[:,0:2,0] -= particles_sim[:,2:4,0]

theta_test = np.array([0.4,0.1,0.05])

#Definition of the movement model for a particle distribution
def movement_model(population, movement, particles,observations,t,dt,theta,rng,num_particles):

    for index,particle in enumerate(particles[:,:,t]):
        particles[index,:,t] = ((SIR_tau_leap(population=population, movement=movement, rng = rng, initial_cond=particle.reshape((3,len(pop_test))).T, theta = theta)[:,:,-1]).T).reshape(-1)
        observations[index,0:len(pop_test),t] = particles[index,len(pop_test):4,t]

    return particles,observations

#PMCMC takes a fixed argument count, thus the closure over population and movement
model = partial(movement_model, pop_test, mov_test)

time = perf_counter()

#Simulate for 60 days
for t in range(1,60):
    particles_sim[:,:,t] = particles_sim[:,:,t-1]
    particles_sim, particle_observations_sim = model(particles = particles_sim,observations=particle_observations_sim,t = t, dt = 1, rng = rng, num_particles=num_particles_sim, theta = theta_test)

print(f"Runtime for {num_particles_sim} particles was {perf_counter() - time} secs")

#Setup the data
data = particle_observations_sim[0,:,:].T #(node,T)

#Run the particle filter
particles, particle_observations,weights,likelihood = filter(data = data, theta = theta_test, num_particles = 100, dt = 1., rng = rng, model = model,model_dim = 6,particle_init = mm_init,observation = mm_obs)

np.savez('PF_Output.npz',distribution = particles, observations = particle_observations, weights = weights, likelihood = likelihood)