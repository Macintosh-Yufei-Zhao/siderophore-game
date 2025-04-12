import numpy as np
import scipy.integrate as ode
import matplotlib.pyplot as plt
import random as rd
from tqdm import tqdm  
import time  

migr = 0                       # migration constant \sigma
# epsilon = np.ones(siderSize)        # siderophore synthesis rate constant
d = 0.1                             # dilution rate constant
u1 = 1
u2 = 1                              # intake rate constant
gamma = 1                           # growth constant

# constants defined: as the original matlab code
supply = 1
# alpha10 = 0.8
# alpha20 = 0.4
# alpha11 = 0.2
# alpha22 = 0.6

def equation(t,z):
    
    m1,m2,r1,r2,iron,v12,v21=z
    
    j1=u1*r1*iron
    j2=u2*r2*iron
    
    r_ratio1=r1/(r1+r2)
    r_ratio2=r2/(r1+r2)
    dv12dt=r_ratio2-v12
    dv21dt=r_ratio1-v21
    
    v11=1-v12
    v22=1-v21
    
    dm1dt= migr + m1 * (gamma * alpha10 * (j1*v11+j2*v12) - d)
    dm2dt= migr + m2 * (gamma * alpha20 * (j1*v21+j2*v22) - d)
    dr1dt= m1 * alpha11 - d * r1
    dr2dt= m2 * alpha22 - d * r2
    dirondt= d * (supply - iron) - m1 * (j1*v11+j2*v12) - m2 * (j1*v21+j2*v22)
    
    return [dm1dt, dm2dt, dr1dt, dr2dt, dirondt, dv12dt, dv21dt]

progress_bar = tqdm(total=101)                          # progress bar, make the simulation progress visible and weaken the boredom...?
 

timespan=(0,2000)
timeeval=np.linspace(0,2000,2001)
m1data=np.zeros((101,101))
m2data=np.zeros((101,101))
for i in range(0,101):
    for j in range(0,101):
        alpha10=i/100
        alpha11=1-alpha10
        alpha20=j/100
        alpha22=1-alpha11
        z01 = [1, 0.01, 1, 1, 1, 0.5, 0.5]
        results=ode.solve_ivp(equation, timespan, z01, t_eval=timeeval)
        m1data[i,j]=results.y[0,2000]
        #z02 = [0.01, 1, 1, 1, 1, 0.5, 0.5]
        #results=ode.solve_ivp(equation, timespan, z02, t_eval=timeeval)
        m2data[i,j]=results.y[1,2000]
    progress_bar.update(1)
progress_bar.close()

log_m1=np.transpose(np.log10(m1data))
log_m2=np.transpose(np.log10(m2data))

#for i in range (0,100):
    #print('m1=',log_m1[0,i],' m2=',log_m2[0,i],'\n')

plt.imshow(log_m1, cmap='coolwarm', interpolation='nearest', aspect='auto',origin='lower')
plt.colorbar()                                                                  # 添加颜色条
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置x轴的刻度
plt.yticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置y轴的刻度
plt.title('Biomass of species 1')
plt.xlabel('alpha10')
plt.ylabel('alpha20')
plt.savefig('2factor-rec-fdbk-1.png')

plt.imshow(log_m2, cmap='coolwarm', interpolation='nearest', aspect='auto',origin='lower')
# plt.colorbar()                                                                  # 添加颜色条
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置x轴的刻度
plt.yticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置y轴的刻度
plt.title('Biomass of species 2')
plt.xlabel('alpha10')
plt.ylabel('alpha20')
plt.savefig('2factor-rec-fdbk-2.png')