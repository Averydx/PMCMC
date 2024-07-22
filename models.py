import numpy as np
import numba as nb

@nb.njit
def OU_model(particles, observations, t, dt, theta, rng,num_particles):
    particles[:,0,t] += -np.exp(theta[0]) * (particles[:,0,t] - theta[1]) * \
    dt + np.sqrt(2 * np.exp(theta[0])) * np.exp(theta[2]) * rng.normal(size = (num_particles,),scale = np.sqrt(dt))

    observations[:,t] = particles[:,0,t]

    return particles,observations

def AR_model(particles, observations, t, dt, theta, rng,num_particles):
    particles[:,t] = theta[0] * particles[:,t-1] + rng.normal(size = (num_particles,),scale = theta[1]) 
    observations[:,t] = particles[:,t] + rng.normal(size = (num_particles,),scale = theta[2])

    return particles, observations   

def SIR_model(particles, t, dt, theta, rng, num_particles):
    pass

