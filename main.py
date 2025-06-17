# -*- coding: utf-8 -*-
"""
Diffuse Raman Simulator (modular version)
"""

import time
import numpy as np
from numpy import sqrt, exp, pi
from scipy.integrate import quad, dblquad, trapz
from scipy.io import savemat
import matplotlib.pyplot as plt

# Import extracted simulation functions (assuming you put them in a module)
from functions import ( A_fit, 
    compute_dR, compute_R, compute_pdf, compute_pdf_cw,
    compute_avg_depth, compute_avg_depth_cw,
    compute_sigma_avg_depth, compute_sigma_avg_depth_cw
)

#%% PARAMETERS of the material. Units -> [cm and s]
mua = 0.1
musp = 10
mur = 10**(-5)
n = 1.4
v = 2.99792*1e10/n
D = 1/(3*musp)
mueff = sqrt(mua/D)

muae = 0.1
muspe = 15
ne = 1.4
ve = 2.99792*1e10/ne
De = 1/(3*muspe)

zs = 1/musp
A = A_fit(1, n)
Ae = A_fit(1, ne)
ze = 2*A*D
zee = 2*Ae*De

#%% PARAMETERS of the simulation
times = np.arange(10, 5000, 100)*1e-12 # [s]
depths = np.arange(0.01, 5.01, 0.01) # [cm]
rhos = np.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7, 2]) # [cm]
fsize = (7,5)

index_rho = [1]
index_times = np.array([1, 9, 19])
index_rhos = np.array([0])
legend_on = True

#%% Compute dR
tic1 = time.time()
dR = compute_dR(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths)
toc1 = time.time()
print("dR computed in", toc1 - tic1, "s")

#%% Compute R
tic2 = time.time()
R = compute_R(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times)
toc2 = time.time()
print("R computed in", toc2 - tic2, "s")

#%% CW case
dR_cw = trapz(dR, times, axis=1)
R_cw = trapz(R, times, axis=1)

#%% Plot dR
plt.figure()
for j in index_rho:        
    for k in index_times:
        plt.plot(depths, dR[j,k], label="rho="+str(rhos[j])+"cm; t="+str(round(times[k]*1e12))+"ps")
if legend_on: plt.legend()
plt.xlabel("depth [cm]")
plt.title("$ dR(z \mid \\rho, t) $ [a.u]")
plt.show()

#%% Plot dR_cw     
plt.figure()        
for k in index_rhos:
    plt.plot(depths, dR_cw[k], label="rho="+str(rhos[k])+"cm")
if legend_on: plt.legend()
plt.xlabel("depth [cm]")
plt.title("$ dR_{CW}(z \mid \\rho) $ [a.u]")
plt.show()

#%% Plot R
plt.figure()
for k in range(len(rhos)):
    plt.plot(times*10**(9), R[k,:,0], label="rho="+str(rhos[k])+"cm")
plt.xlabel('time [ns]')
plt.ylabel('$ R_e(\\rho,t) $ [a.u.]')
if legend_on: plt.legend()
plt.show()

#%% Compute PDF
prob = compute_pdf(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths)
prob_cw = compute_pdf_cw(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths)

#%% Plot PDF time-domain
plt.rcParams.update({'font.size': 16})
from matplotlib.ticker import StrMethodFormatter

plt.figure(figsize=fsize)
for j in index_rho:        
    for k in index_times:
        plt.plot(depths, prob[j,k], label="$\\rho$="+str(rhos[j])+"cm; t="+str(round(times[k]*1e12))+"ps")
if legend_on: plt.legend()
plt.ylabel("$f \, (z \mid t)$")
plt.xlabel("depth [cm]")
plt.title('probability density function (time-domain)')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
plt.xlim((0,3))
plt.ylim(bottom = 0)
plt.grid(True)
plt.show()

checkNorm = trapz(prob, depths)
print("Normalization check f(z,t) =", checkNorm)

#%% Plot PDF CW
plt.figure(figsize=fsize)        
for k in index_rhos:
    plt.plot(depths, prob_cw[k], label="$\\rho$="+str(rhos[k])+"cm")
if legend_on: plt.legend()
plt.ylabel("$f \, (z \mid \\rho)$")
plt.xlabel("depth [cm]")
plt.title('probability density function (CW case)')
plt.xlim((0,3))
plt.ylim((0,4))
plt.grid(True)
plt.show()

checkNorm_cw = trapz(prob_cw, depths)
print("Normalization check f(z,rho) =", checkNorm_cw)

#%% Compute and plot average depth
avg_depth = compute_avg_depth(prob, depths)
plt.figure(figsize=fsize)
for k in index_rho:
    plt.plot(times*1e9, avg_depth[k], label="$\\rho$="+str(rhos[k])+"cm", color='blue')
plt.xlabel('t [ns]')
plt.ylabel('$z_{Raman}$ [cm]')
plt.title('average generation depth (time-domain)')
plt.xlim((0, times[-1]*1e9))
plt.ylim(bottom = 0)
plt.grid(True)
plt.show()

avg_depth_cw = compute_avg_depth_cw(prob_cw, depths)
plt.figure(figsize=fsize)
plt.plot(rhos, avg_depth_cw, color='blue')
plt.xlabel('$\\rho$ [cm]')
plt.ylabel('$z_{Raman, CW}$ [cm]')
plt.title('average generation depth (CW case)')
plt.xlim((0, rhos[-1]))
plt.ylim(bottom = 0)
plt.grid(True)
plt.show()

#%% Compute and plot standard deviation
sigma_avg_depth = compute_sigma_avg_depth(prob, depths, avg_depth)
plt.figure(figsize=fsize)
for k in index_rho:
    plt.plot(times*1e9, avg_depth[k], label="rho="+str(rhos[k])+"cm", color='blue')
    lower_bound = avg_depth[k] - sigma_avg_depth[k]
    lower_bound[lower_bound < 0] = 0
    upper_bound = avg_depth[k] + sigma_avg_depth[k]
    plt.fill_between(times*1e9, lower_bound, upper_bound, alpha=0.1)
plt.xlabel('t [ns]')
plt.ylabel('$ z_{Raman}$ [cm]')
plt.title('st.dev. of zRaman (time-domain)')
plt.xlim((0, times[-1]*1e9))
plt.ylim(bottom = 0)
plt.grid(True)
plt.show()

sigma_avg_depth_cw = compute_sigma_avg_depth_cw(prob_cw, depths, avg_depth_cw)
plt.figure(figsize=fsize)
plt.plot(rhos, avg_depth_cw, color='blue')
lower_bound_cw = avg_depth_cw - sigma_avg_depth_cw
lower_bound_cw[lower_bound_cw < 0] = 0
upper_bound_cw = avg_depth_cw + sigma_avg_depth_cw
plt.fill_between(rhos, lower_bound_cw, upper_bound_cw, alpha=0.1)
plt.xlabel('$\\rho$ [cm]')
plt.ylabel('$z_{Raman, CW}$ [cm]')
plt.title('st.dev. of zRaman (CW case)')
plt.xlim((0, rhos[-1]))
plt.ylim(bottom = 0)
plt.grid(True)
plt.show()

#%% Save results
METHOD = 4
mdict = {
    'mua':mua, 'muae':muae, 'musp':musp, 'muspe':muspe, 'mur':mur,
    'rhos':rhos, 'n':n, 'A':A, 'zs':zs, 'method':METHOD,
    'pdf':prob, 'pdf_cw':prob_cw, 'avg_depth':avg_depth,
    'avg_depth_cw':avg_depth_cw, 'time':times*1e12, 'depths':depths
}
savemat('C:/Users/Valerio/Desktop/DiffuseRaman/ModelData/pdf_8_model.mat', mdict)

