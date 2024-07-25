import numpy as np
import numba as nb

@nb.njit
def OU_model(particles, observations, t, dt, theta, rng,num_particles):
    particles[:,0,t] += -np.exp(theta[0]) * (particles[:,0,t] - theta[1]) * \
    dt + np.sqrt(2 * np.exp(theta[0])) * np.exp(theta[2]) * rng.normal(size = (num_particles,),scale = np.sqrt(dt))

    observations[:,t] = particles[:,0,t]

    return particles,observations

@nb.njit
def AR_model(particles, observations, t, dt, theta, rng,num_particles):
    particles[:,0,t] = theta[0] * particles[:,0,t] + rng.normal(size = (num_particles,),scale = theta[1]) 
    observations[:,t] = particles[:,0,t] + rng.normal(size = (num_particles,),scale = theta[2])

    return particles, observations   

@nb.njit
def AR_PMCMC_model(particles, observations, t, dt, theta, rng,num_particles):
    particles[:,0,t] = theta[0] * particles[:,0,t] + rng.normal(size = (num_particles,),scale = np.exp(theta[1])) 
    observations[:,t] = particles[:,0,t] + rng.normal(size = (num_particles,),scale = np.exp(theta[2]))

    return particles, observations

def logistic_model(particles,observations,t,dt,theta,rng,num_particles):
    particles[:,0:2,t] += particles[:,2:,t] * particles[:,0:2,t] * (1 - particles[:,0:2,t]/theta) * dt

    lam = 1.0/365
    m= 0.0
    sig = 0.3

    particles[:,2,t] += -lam * (particles[:,2,t] - m) * dt + np.sqrt(2 * lam) * sig * rng.normal(0,scale = np.sqrt(dt))

    observations[:,:,t] = particles[:,0:2,t]

    return particles,observations



#r(t) is the growth rate of the population at time t
#max growth rate is 0.3 with time period of 365 days
def r_t(t):
    num_models = 2
    period = 365/2.0
    d=365
    return 0.10*np.sin(2*np.pi*t/period)