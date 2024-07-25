import numpy as np
import numba as nb
from numpy.linalg import cholesky,LinAlgError
from kalman_filter import filter

def KMCMC(iterations, num_particles, init_theta, prior, model, data, rng, dt,model_dim): 

    theta = np.zeros((len(init_theta),iterations))
    LL = np.zeros((iterations,))

    mu = np.zeros((len(init_theta),))
    L = np.eye(len(init_theta))

    theta[:,0] = init_theta

    LL[0] = prior(init_theta) 

    if(np.isfinite(LL[0])):
        likelihood = filter(data = data,theta= theta[:,0] ,rng = rng)
        LL[0] += np.sum(likelihood)

    #create a zero vector to store the acceptance rate
    acc_record = np.zeros((iterations,))

    '''PMCMC Loop'''
    for iter in range(1,iterations): 
        
        if(iter % 10 == 0):
            #print the acceptance rate and likelihood every 10 iterations
            print(f"iteration: {iter}" + f"| Acceptance rate: {np.sum(acc_record[:iter])/iter}" + f"| Log-Likelihood: {LL[iter-1]}")

        # z = rng.standard_normal(size = (len(init_theta)))
        # theta_prop = theta[:,iter - 1] + np.dot(L, z)

        if iter > 100:
            theta_prop = rng.multivariate_normal(theta[:, iter-1],cov = 2.38/np.sqrt(len(init_theta)) * np.cov(theta))

        else:  
            theta_prop = rng.multivariate_normal(theta[:, iter-1],cov = np.eye(np.shape(theta)[0]))


        LL_new = prior(theta_prop)

        if(np.isfinite(LL_new)):
            likelihood = filter(data = data,theta= theta_prop,rng = rng)
            LL_new += np.sum(likelihood)

        ratio = (LL_new - LL[iter-1])

            ###Random number for acceptance criteria
        u = rng.uniform(0.,1.)
        if u < min(1,np.exp(ratio)): 
            #print(f"accepted proposal {theta_prop} w/ Log-Likelihood {LL_new}")
            theta[:,iter] = theta_prop
            LL[iter] = LL_new
            acc_record[iter] = 1
        else: 
            theta[:,iter] = theta[:,iter - 1]
            LL[iter] = LL[iter-1]

        #mu, L = cov_update(L,mu,theta[:,iter],iter)

    return theta,LL

def cov_update(cov, mu, theta_val,iteration):
    g = (iteration + 1) ** (-0.6)
    mu = (1. - g) * mu + g * theta_val

    m_theta = theta_val - mu

    cov = (1. - g) * cov + g * np.outer(m_theta,m_theta.T)
    cov = cov * 2.38/np.sqrt(len(theta_val))

    try:
        L = cholesky(cov)

    except LinAlgError:
        L = np.eye(3)

    return mu,L

