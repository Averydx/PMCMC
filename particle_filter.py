import numpy as np
import numba as nb

@nb.njit
def filter(data,theta,num_particles,dt,rng,model,model_dim):
    '''Initialize the particle distribution'''

    particles = np.zeros((num_particles,model_dim,len(data)),dtype = np.float_)

    #TODO Assumption here that observations are 1-D, will definitely need to change for the movement model
    particle_observations = np.zeros((num_particles,len(data)),dtype=np.float_)

    particles[:,0,0] = rng.normal(size=(num_particles * model_dim,))

    weights = np.zeros((num_particles,len(data)),dtype = np.float_)
    likelihood = np.zeros((len(data),),dtype=np.float_)

    for t,data_point in enumerate(data):

        '''Simulation/forecast step for all t > 0'''
        if(t > 0):
            particles,particle_observations = simulate(particles=particles,particle_observations=particle_observations,t = t,dt = dt,theta = theta,model = model,rng = rng,num_particles=num_particles)
        '''Resampling and weight computation'''
        ob_var = 1.**2
        weights[:,t] = 1/np.sqrt(2 * np.pi * ob_var) * np.exp(-((data_point - particle_observations[:,t])**2)/(2 * ob_var))

        likelihood[t] = np.mean(weights[:,t])

        weights[:,t] = (weights[:,t] / np.sum(weights[:,t])) #Normalization

        particles[:,:,t] = resampling(particles[:,:,t],weights[:,t],rng)

    return particles,weights,likelihood

@nb.njit
def resampling(particles,weights,rng): 
    indices = np.zeros(len(weights),dtype = np.int_) #initialize array to hold the indices
    cdf = np.cumsum(weights) #create cdf

    u = rng.uniform(0,1/len(weights)) #random number between 1 and 1/n, only drawn once vs the n 
    i = 0
    for j in range(0,len(weights)): 
        r = (u + 1/len(weights) * j)
        while r > cdf[i]: 
            i += 1
        indices[j] = i

    return particles[indices,:]

@nb.njit
def simulate(particles, particle_observations,t,dt,theta,model,rng,num_particles):      
    particles[:,:,t] = particles[:,:,t-1]
    for _ in range(int(1/dt)):
        particles,particle_observations = model(particles, particle_observations, t, dt, theta, rng,num_particles)

    return particles,particle_observations

        