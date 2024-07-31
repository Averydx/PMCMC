import numpy as np
import numba as nb
from particle_filter import filter
from numpy.linalg import cholesky,LinAlgError

def PMCMC(iterations, num_particles, init_theta, prior, model, observation, data, rng, dt,model_dim,observation_dim = 1): 

    pf_means = np.zeros((model_dim,len(data),iterations))

    theta = np.zeros((len(init_theta),iterations))
    LL = np.zeros((iterations,))

    mu = np.zeros(len(init_theta))
    cov = np.eye(len(init_theta))

    theta[:,0] = init_theta
    LL[0] = prior(init_theta) 

    if(np.isfinite(LL[0])):
        particles,weights,likelihood = filter(data = data,
                                theta= theta[:,0],
                                rng = rng,num_particles = num_particles,
                                dt = dt, 
                                model = model,
                                observation=observation,
                                model_dim=model_dim,
                                observation_dim=observation_dim)
        
        pf_means[:,:,0] = pf_mean_calc(particles,model_dim,weights)
        LL[0] += np.sum(np.log(likelihood))

    #create a zero vector to store the acceptance rate
    acc_record = np.zeros((iterations,))

    '''PMCMC Loop'''

    for iter in range(1,iterations): 
        
        if(iter % 10 == 0):
            #print the acceptance rate and likelihood every 10 iterations
            print(f"iteration: {iter}" + f"| Acceptance rate: {np.sum(acc_record[:iter])/iter}" + f"| Log-Likelihood: {LL[iter-1]}" + f"| Proposal {theta[:,iter - 1]}")

        z = rng.standard_normal((len(theta[:,iter-1])))
        L = cholesky((2.38**2/len(theta[:,iter - 1])) * cov) 
        theta_prop = theta[:,iter - 1] + L @ z

        LL_new = prior(theta_prop)

        if(np.isfinite(LL_new)):
            particles,weights,likelihood = filter(data = data,
                                    theta= theta_prop,
                                    rng = rng,
                                    num_particles = num_particles,
                                    dt = dt,
                                    model = model,
                                    observation=observation,
                                    model_dim=model_dim,
                                    observation_dim=observation_dim)
            
            pf_means[:,:,iter] = pf_mean_calc(particles,model_dim,weights)
            
            LL_new += np.sum(np.log(likelihood))

        ratio = (LL_new - LL[iter-1])

            ###Random number for acceptance criteria
        u = rng.uniform(0.,1.)
        if np.log(u) < ratio: 
            theta[:,iter] = theta_prop
            LL[iter] = LL_new
            acc_record[iter] = 1
        else: 
            theta[:,iter] = theta[:,iter - 1]
            LL[iter] = LL[iter-1]

        mu, cov = cov_update(cov,mu,theta[:,iter],iter)

    return theta,LL,pf_means 



@nb.njit
def cov_update(cov, mu, theta_val,iteration):

    g = (iteration + 1) ** (-0.6)
    mu = (1.0 - g) * mu + g * theta_val
    m_theta = theta_val - mu
    cov = (1.0 - g) * cov + g * np.outer(m_theta,m_theta.T)

    return mu,cov

def pf_mean_calc(particles,model_dim,weights):
    means = np.zeros((model_dim,weights.shape[1]))
    for axis in range(model_dim):
        means[axis,:] = np.average(particles[:,axis,:],weights=weights,axis = 0)
        
    return means