import numpy as np
import scipy.integrate as ode
import matplotlib.pyplot as plt
import random as rd
import os

migr = 0                       # migration constant \sigma
# epsilon = np.ones(siderSize)        # siderophore synthesis rate constant
d = 0.1                             # dilution rate constant
u1 = 1
u2 = 1                              # intake rate constant
gamma = 1                           # growth constant

# constants defined: as the original matlab code
supply = 1
alpha10 = 0.8
alpha20 = 0.4
alpha11 = 0.2
alpha22 = 0.6

def equation(t,z):
    
    m1,m2,r1,r2,iron=z
    
    j1=u1*r1*iron
    j2=u2*r2*iron
    
    dm1dt= migr + m1 * (gamma * alpha10 * (j1*v11+j2*v12) - d)
    dm2dt= migr + m2 * (gamma * alpha20 * (j1*v21+j2*v22) - d)
    dr1dt= m1 * alpha11 - d * r1
    dr2dt= m2 * alpha22 - d * r2
    dirondt= d * (supply - iron) - m1 * (j1*v11+j2*v12) - m2 * (j1*v21+j2*v22)
    
    return [dm1dt, dm2dt, dr1dt, dr2dt, dirondt]

timespan=(0,2000)
timeeval=np.linspace(0,2000,2001)
m1maj1data=np.zeros((101,101))
m2maj1data=np.zeros((101,101))
m1maj2data=np.zeros((101,101))
m2maj2data=np.zeros((101,101))
for i in range(0,101):
    for j in range(0,101):
        v11=i/100
        v12=1-v11
        v22=j/100
        v21=1-v22
        z01 = [1, 0.01, 1, 1, 1]
        results=ode.solve_ivp(equation, timespan, z01, t_eval=timeeval)
        m1maj1data[i,j]=results.y[0,2000]
        m2maj1data[i,j]=results.y[1,2000]
        z02 = [0.01, 1, 1, 1, 1]
        results=ode.solve_ivp(equation, timespan, z02, t_eval=timeeval)
        m1maj2data[i,j]=results.y[0,2000]
        m2maj2data[i,j]=results.y[1,2000]
    # print("\n")

log_m1maj1=np.log10(m1maj1data)
log_m1maj2=np.log10(m1maj2data)
log_m2maj1=np.log10(m2maj1data)
log_m2maj2=np.log10(m2maj2data)

savepath='241230'
os.makedirs(savepath)

# the following graphing is included in graphing.py
plt.imshow(log_m1maj1, cmap='coolwarm', interpolation='nearest', aspect='auto',origin='lower')
plt.colorbar()                                                                  # 添加颜色条
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置x轴的刻度
plt.yticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置y轴的刻度
plt.title('Biomass of species 1 with 1 major')
plt.xlabel('v11')
plt.ylabel('v22')
plt.savefig(savepath+'/2factor-m1maj1.png')
plt.close()

plt.imshow(log_m2maj1, cmap='coolwarm', interpolation='nearest', aspect='auto',origin='lower')
plt.colorbar()                                                                  # 添加颜色条
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置x轴的刻度
plt.yticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置y轴的刻度
plt.title('Biomass of species 2 with 1 major')
plt.xlabel('v11')
plt.ylabel('v22')
plt.savefig(savepath+'/2factor-m2maj1.png')
plt.close()

plt.imshow(log_m1maj2, cmap='coolwarm', interpolation='nearest', aspect='auto',origin='lower')
plt.colorbar()                                                                  # 添加颜色条
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置x轴的刻度
plt.yticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置y轴的刻度
plt.title('Biomass of species 1 with 2 major')
plt.xlabel('v11')
plt.ylabel('v22')
plt.savefig(savepath+'/2factor-m1maj2.png')
plt.close()

plt.imshow(log_m2maj2, cmap='coolwarm', interpolation='nearest', aspect='auto',origin='lower')
plt.colorbar()                                                                  # 添加颜色条
plt.xticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.6', '0.8', '1.0'])  # 设置x轴的刻度
plt.yticks([0, 20, 40, 60, 80, 100], ['0', '0.2', '0.4', '0.python two_species_redo.pypython two_species_redo.py6', '0.8', '1.0'])  # 设置y轴的刻度
plt.title('Biomass of species 2 with 2 major')
plt.xlabel('v11')
plt.ylabel('v22')
plt.savefig(savepath+'/2factor-m2maj2.png')
plt.close()