import numpy as np
import scipy.integrate as ode
import matplotlib.pyplot as plt

# parameters

migr = 1e-8                         # migration constant \sigma
epsilon = np.ones(siderSize)        # siderophore synthesis rate constant
d = 0.1                             # dilution rate constant
u = np.ones(siderSize)              # intake rate constant
gamma = 1                           # growth constant

# ODE

def total(t,z) :
    v, m, r, iron = z 
    
    # update flux j
    j = np.zeros(siderSize)
    for i in range (0, siderSize):
        j[i] = u[i] * r[i] * iron
    
    # normalization of v
    vnew = np.zeros((speciesSize, siderSize))
    for i in range (0, speciesSize):
        sum = 0
        for k in range (0, siderSize):
            sum += v[i,k]
        for k in range (0, siderSize):
            vnew[i,k] = v[i,k] / sum
    v = vnew
    
    # ratio of siderophores
    r_ratio = np.zeros(siderSize)
    sum = 0
    for i in range(0,siderSize):
        sum += r[i]
    for i in range(0,siderSize):
        r_ratio[i]=r[i]/sum
    
    # update receptor strategy v
    dvdt = np.zeros((speciesSize, siderSize))
    for i in range (0, speciesSize):
        for k in range (0, siderSize):
            dvdt[i,k] = r[k]-v[i,k]
    
    # update biomass m
    dmdt = np.zeros(speciesSize)
    for i in range (0, speciesSize):
        sum = 0
        for k in range (0, siderSize):
            sum += j[k] * v[k]
        dmdt[i] = migr + m[i] * (gamma * alpha[i, 0] * sum - d)
    
    # update siderophore concentration r
    drdt = np.zeros(siderSize)
    for i in range(0, siderSize):
        sum = 0
        for k in range (0, speciesSize):
            sum += m[k] * alpha[k,j] * epsilon[j]
        drdt[i] = sum - d * r[i]
    
    # update iron 
    sum = 0
    for i in range (0, speciesSize):
        for k in range (0, siderSize):
            sum += m[i] * v[i,k] * j[k]
    dirondt = d * (supply - iron) - sum
    
    return [dvdt, dmdt, drdt, dirondt]



# input

speciesSize = 2                     # N_spe
siderSize = 2                       # N_sid

alpha = np.zeros((speciesSize,siderSize+1))             # resource assignment

v = np.zeros((speciesSize,siderSize))                   # receptor assignment

supply = 1





# initial states

z0=
timespan=(0,100)
# time_eval=np.linspace(0,100,1000)

solution = ode.solve_ivp(total, timespan, z0)



