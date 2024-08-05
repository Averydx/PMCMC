import numpy as np
from numpy.typing import NDArray
import numba as nb
from scipy.stats import norm

def filter(data,theta,num_particles,dt,rng,model,observation,model_dim,particle_init):
    '''Initialize the particle distribution'''

    particles = np.zeros((num_particles,model_dim,len(data)),dtype = np.float64)

    #TODO Assumption here that observations are 1-D, will definitely need to change for the movement model
    particle_observations = np.zeros((num_particles,data.shape[1],data.shape[0]),dtype=np.float64)

    #particles[:,:,0] = np.ones(shape=(num_particles, model_dim,))

    compartment_init = rng.integers(0,5,size = (num_particles,2))

    particles[:,0,0] = 100_000 - np.sum(compartment_init,axis = 1)
    particles[:,1,0] = compartment_init[:,0]
    particles[:,2,0] = compartment_init[:,1]
    particles[:,4,0] = rng.uniform(0.2,0.5,size = (num_particles,))


    weights = np.zeros((num_particles,len(data)),dtype = np.float64)
    likelihood = np.zeros((len(data),),dtype=np.float64)

    for t,data_point in enumerate(data):

        '''Simulation/forecast step for all t > 0'''
        if(t > 0):
            particles,particle_observations = simulate(particles=particles,
                                                       particle_observations=particle_observations,
                                                       t = t,
                                                       dt = dt,
                                                       theta = theta
                                                       ,model = model,
                                                       rng = rng,
                                                       num_particles=num_particles)
            
        '''Resampling and weight computation'''
        weights[:,t] = observation(data_point = data_point,
                                   particle_observations = particle_observations[:,:,t],
                                   theta = theta)

        likelihood[t] = jacob(weights[:,t])[-1] - np.log(num_particles)

        weights[:,t] = log_norm(weights[:,t]) #/ np.sum(weights[:,t]) #Normalization

        particles[:,:,t] = log_resampling(particles[:,:,t],weights[:,t],rng)

    return particles,particle_observations,weights,likelihood

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
def log_resampling(particles,weights,rng): 
    indices = np.zeros(len(weights),dtype = np.int_) #initialize array to hold the indices
    cdf = jacob(weights) #create cdf

    u = rng.uniform(0,1/len(weights)) #random number between 1 and 1/n, only drawn once vs the n 
    i = 0
    for j in range(0,len(weights)): 
        r = np.log(u + 1/len(weights) * j)
        while r > cdf[i]: 
            i += 1
        indices[j] = i

    return particles[indices,:]

def simulate(particles, particle_observations,t,dt,theta,model,rng,num_particles):      
    particles[:,:,t] = particles[:,:,t-1]
    for _ in range(int(1/dt)):
        particles,particle_observations = model(particles, particle_observations, t, dt, theta, rng,num_particles)

    return particles,particle_observations

@nb.njit
def jacob(δ:NDArray[np.float_])->NDArray[np.float_]:
    """The jacobian logarithm, used in log likelihood normalization and resampling processes
    δ will be an array of values. 
    
    Args: 
        δ: An array of values to sum

    Returns: 
        The vector of partial sums of δ.          
    
    """
    n = len(δ)
    Δ = np.zeros(n)
    Δ[0] = δ[0]
    for i in range(1,n):
        Δ[i] = max(δ[i],Δ[i-1]) + np.log(1 + np.exp(-1*np.abs(δ[i] - Δ[i-1])))
    return(Δ)

@nb.njit
def log_norm(log_weights:NDArray): 
    '''normalizes the probability space using the jacobian logarithm as defined in jacob() '''
    norm = (jacob(log_weights)[-1])
    log_weights -= norm
    return log_weights
