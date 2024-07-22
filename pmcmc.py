import numpy as np
import numba as nb
from particle_filter import filter

def PMCMC(iterations, num_particles, init_theta, prior, model, data, rng, dt,model_dim,burn_in = 1000): 

    theta = np.zeros((len(init_theta),iterations))
    LL = np.zeros((iterations,))

    theta[:,0] = init_theta

    _,_,likelihood = filter(data = data,theta= theta[:,0] ,rng = rng,num_particles = num_particles,dt = dt, model = model,model_dim=model_dim)

    LL[0] = np.sum(np.log(likelihood))

    '''PMCMC Loop'''
    for iter in range(1,iterations): 
        
        if(iter % 10 == 0):
            print(f"iteration: {iter}")

        if iter > burn_in:
            cov = 2.38/np.shape(theta)[0] * np.cov(theta)
            theta_prop = rng.multivariate_normal(theta[:, iter-1],cov = cov)

        else:  
            theta_prop = rng.multivariate_normal(theta[:, iter-1],cov = 0.1 * np.eye(np.shape(theta)[0]))

        _,_,likelihood = filter(data = data,theta= theta_prop,rng = rng,num_particles = num_particles,dt = dt,model = model,model_dim=model_dim)

        LL_new = np.sum(np.log(likelihood))

        ratio = (LL_new - LL[iter-1]) + (prior(theta_prop) - prior(theta[:,iter-1]))

            ###Random number for acceptance criteria
        u = rng.uniform(0.,1.)
        if(u < min(1,np.exp(ratio))): 
            #print(f"accepted proposal {theta_prop} w/ Log-Likelihood {LL_new}")
            theta[:,iter] = theta_prop
            LL[iter] = LL_new
        else: 
            theta[:,iter] = theta[:,iter - 1]
            LL[iter] = LL[iter-1]

    return theta,LL

