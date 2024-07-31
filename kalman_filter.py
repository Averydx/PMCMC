import numpy as np
import numba as nb
from scipy.stats import norm

def filter(data,theta,rng):
    
    '''Implementation of the Kalman Filter'''

    posterior_mean = np.zeros(len(data))
    posterior_cov = np.zeros(len(data))

    posterior_mean[0] = 0
    posterior_cov[0] = 1

    LL = np.zeros(len(data))

    for time_index in range(len(data)):

        '''Forecast'''
        if(time_index > 0):
            posterior_mean[time_index] = theta[0] * posterior_mean[time_index-1] + rng.normal(0,theta[1])
            posterior_cov[time_index] = theta[0] * posterior_cov[time_index-1] * theta[0] + theta[1]**2

        '''Likelihood'''
        LL[time_index] = norm.logpdf(x = data[time_index], loc = posterior_mean[time_index],scale = np.sqrt(posterior_cov[time_index] + theta[2]**2))

        '''Update'''
        K = posterior_cov[time_index] / (posterior_cov[time_index] + theta[2]**2)

        posterior_mean[time_index] = posterior_mean[time_index] + K * (data[time_index] - posterior_mean[time_index])
        posterior_cov[time_index] = (1 - K) * posterior_cov[time_index]

    return LL