import numpy as np
from numpy.typing import NDArray
import numba as nb
from scipy.stats import norm

def filter(data,theta,num_particles,dt,rng,model,observation,model_dim,particle_init):
    '''Initialize the particle distribution, observations and weights. 
    
    Args: 
        data: A (observation_dim,T) matrix of observations of the system. 
        theta: Vector of system parameters, used in the observation density and transition density. 
        num_particles: How many particles to use to perform inference. 
        dt: Discretization step of a continuous time model, for discrete SSMs set to 1.  
        rng: An instance of the NumPy Generator class. Used for random number generation. 
        model: A python function describing the transition map for the model. Arguments are (particles,observations,t,dt,theta,rng,num_particles)
        observation_model: A python function describing the observation density/measure. Arguments are (data_point, particle_observations, theta)
        model_dim: dimension of the model 
        particle_init: Initializer function for the particles.

    Returns: 
        The vector of partial sums of δ.  

    '''

    particles = np.zeros((num_particles,model_dim,len(data)),dtype = np.float64)

    particle_observations = np.zeros((num_particles,data.shape[1] if len(data.shape ) > 1 else 1,data.shape[0]),dtype=np.float64)

    particles[:,:,0] = particle_init(num_particles,model_dim,rng)

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

        likelihood[t] = jacob(weights[:,t])[-1] - np.log(num_particles) # Computes the Monte Carlo estimate of the likeihood. I.E. P(y_{1:T})

        weights[:,t] = log_norm(weights[:,t]) #/ np.sum(weights[:,t]) #Normalization step

        particles[:,:,t] = log_resampling(particles[:,:,t],weights[:,t],rng) #Resampling the particles

    return particles,particle_observations,weights,likelihood

@nb.njit
def resampling(particles,weights,rng): 

    '''Systematic resampling algorithm, the njit decorator is important here as it gives a significant speedup. Time 
    complexity is O(n), as opposed to O(nlog(n)) in multinomial resampling. '''

    indices = np.zeros(len(weights),dtype = np.int_) #initialize array to hold the indices
    cdf = np.cumsum(weights) #create cdf

    u = rng.uniform(0,1/len(weights)) #random number between 1 and 1/n, only drawn once vs the n draws in multinomial resampling
    i = 0
    for j in range(0,len(weights)): 
        r = (u + 1/len(weights) * j)
        while r > cdf[i]: 
            i += 1
        indices[j] = i

    return particles[indices,:]

@nb.njit
def log_resampling(particles,weights,rng): 

    '''Systematic resampling algorithm in log domain, the njit decorator is important here as it gives a significant speedup. Time 
    complexity is O(n), as opposed to O(nlog(n)) in multinomial resampling. '''

    indices = np.zeros(len(weights),dtype = np.int_) #initialize array to hold the indices
    cdf = jacob(weights) #create log-cdf

    u = rng.uniform(0,1/len(weights)) #random number between 1 and 1/n, only drawn once vs the n draws in multinomial resampling
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
def jacob(δ:NDArray[np.float64])->NDArray[np.float64]:
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
