import numpy as np
import numba as nb
from scipy.stats import norm

'''OU Implementations'''
@nb.njit
def OU_model(particles,observations,t,dt,theta,rng):
    particles[:,0,t] += -theta[0] * (particles[:,0,t] - theta[1]) * \
    dt + np.sqrt(2 * theta[0]) * theta[2] * rng.normal(size = (particles.shape[0],),scale = np.sqrt(dt))

    observations[:,0,t] = particles[:,0,t]
    return particles,observations

@nb.njit
def OU_Obs(data_point, particle_observations, theta):
    return 1/np.sqrt(2 * np.pi * theta[3]**2) * np.exp(-((data_point - particle_observations[:,0])**2)/(2 * theta[3] ** 2))


'''AR Implementations'''
@nb.njit
def AR_model(particles, observations, t, dt, theta, rng):
    particles[:,0,t] = theta[0] * particles[:,0,t] + rng.normal(size = (particles.shape[0],),scale = theta[1]) 
    observations[:,0,t] = particles[:,0,t]

    return particles, observations   

@nb.njit 
def AR_Obs(data_point, particle_observations, theta):
    return 1/np.sqrt(2 * np.pi * theta[2]**2) * np.exp(-((data_point - particle_observations[:,0])**2)/(2 * theta[2] ** 2))

def SEIR_model(particles,observations,t,dt,theta,rng):

    gamma,eta = theta

    new_E = rng.poisson((particles[:,4,t] * (particles[:, 1, t] + 0.1 * particles[:, 2, t]) * particles[:, 0, t])/np.sum(particles[:,:,t],axis = 1) * dt)
    new_I = rng.poisson((eta * particles[:,1,t]) * dt)
    new_ER = rng.poisson((gamma * particles[:,1,t]) * dt)
    new_IR = rng.poisson((gamma * particles[:,2,t]) * dt)
    new_D = rng.poisson((0.004 * particles[:,2,t]) * dt)

    sig = 1. 
    lam = 1/365
    mu = -0.5

    A = np.exp(-lam * dt)
    M = mu * (np.exp(-lam * dt) - 1)
    C = sig * np.sqrt(1 - np.exp(-2 * lam * dt))

    # new_E = (particles[:,4,t] * (particles[:, 1, t] + 0.1 * particles[:, 2, t]) * particles[:, 0, t])/np.sum(particles[:,:,t],axis = 1) * dt
    # new_I = (eta * particles[:,1,t]) * dt
    # new_ER = (gamma * particles[:,1,t]) * dt
    # new_IR = (gamma * particles[:,2,t]) * dt
    # new_D = (0.004 * particles[:,2,t]) * dt

    particles[:,0,t] = np.maximum(0.,particles[:,0,t] - new_E) #S
    particles[:,1,t] = np.maximum(0.,particles[:,1,t] + new_E - new_I - new_ER) #E
    particles[:,2,t] = np.maximum(0.,particles[:,2,t] + new_I - new_IR - new_D) #I
    particles[:,3,t] = np.maximum(0.,particles[:,3,t] + new_ER + new_IR)
    particles[:,4,t] = np.exp(A * np.log(particles[:,4,t]) - M + C * rng.standard_normal(size = (particles.shape[0],)))#beta_sim(beta_par,t)#

    observations[:,0,t] = particles[:,2,t]#new_E#particles[:,1,t]

    return particles,observations

def SEIR_Obs(data_point, particle_observations, theta):
    return norm.logpdf(data_point, particle_observations[:,0],scale = 15)

beta_par = {'b_0':0.4,'b_inf': 0.1, 'tau': 5,'T':20}

def beta_sim(par,t):
    if(t < par['T']):
        return 0.4
    
    return par['b_inf'] + (par['b_0'] - par['b_inf']) * np.exp(-(t - par['T'])/par['tau'])

