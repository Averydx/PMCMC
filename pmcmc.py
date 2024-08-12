import numpy as np
import numba as nb
from particle_filter import filter
from numpy.linalg import cholesky,LinAlgError

def PMCMC(iterations, num_particles, init_theta, prior, model, observation, data, rng, dt,model_dim, particle_init): 

    MLE_Particles = np.zeros((num_particles,model_dim,len(data)))
    MLE_Observations = np.zeros((num_particles,data.shape[1] if len(data.shape ) > 1 else 1,data.shape[0]),dtype=np.float64)

    MLE = -50000

    theta = np.zeros((len(init_theta),iterations))
    LL = np.zeros((iterations,))

    mu = np.zeros(len(init_theta))
    cov = np.eye(len(init_theta))

    theta[:,0] = init_theta
    LL[0] = prior(init_theta) 

    if(np.isfinite(LL[0])):
        particles,particle_observations,weights,likelihood = filter(data = data,
                                theta= theta[:,0],
                                rng = rng,num_particles = num_particles,
                                dt = dt, 
                                model = model,
                                observation=observation,
                                model_dim=model_dim,
                                particle_init=particle_init)
        

        LL[0] += np.sum(likelihood)

        MLE = LL[0]
        MLE_Particles = particles
        MLE_Observations = particle_observations

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
            particles,particle_observations,weights,likelihood = filter(data = data,
                                    theta= theta_prop,
                                    rng = rng,
                                    num_particles = num_particles,
                                    dt = dt,
                                    model = model,
                                    observation=observation,
                                    model_dim=model_dim,
                                    particle_init=particle_init)
            
            
            
            LL_new += np.sum((likelihood))

            if(LL_new > MLE):
                MLE = LL_new
                MLE_Particles = particles
                MLE_Observations = particle_observations

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

    return theta,LL,MLE_Particles,MLE_Observations



@nb.njit
def cov_update(cov, mu, theta_val,iteration):

    g = (iteration + 1) ** (-0.6)
    mu = (1.0 - g) * mu + g * theta_val
    m_theta = theta_val - mu
    cov = (1.0 - g) * cov + g * np.outer(m_theta,m_theta.T)

    return mu,cov
