import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import constants
from scipy.optimize import curve_fit

k = constants.k             # J/K
hbar = constants.hbar       # J*s
m_e = constants.m_e         # kg
mu_0 = constants.mu_0       # N/A^2
elm = constants.e           # C
mu_B = elm*hbar/(2*m_e)     # A*m^2
g = 2.0023318416            # Dimensionless
mol = constants.N_A

df = pd.read_excel('SQUID_raw_data.xlsx', header=None) #SQUID_measurements for fit, # SQUID_raw_data for raw data
data = df.to_numpy().T
exp_T, exp_chiT = data[:][0], data[:][1] # Dimensionless, K
exp_chiT = exp_chiT * 4 * np.pi * 1e-6 # -> to SI

def bleaney_bower(T, J2, g, C, x):
    chiT = (2*mol*mu_0*g**2*mu_B**2)/(k*(3+np.exp(-J2/(k*T)))) + C + x*T
    return chiT

#print(60 <= exp_T <= 235) or (63 <= exp_T <= 245)
maskt1 = exp_T <= 245
maskt2 = exp_T >= 63
maskt3 = maskt1*maskt2

fit_T = exp_T[maskt3]
fit_chiT = exp_chiT[maskt3]

bounds = ([-0.025*1.602e-19, g, -2, -2], [0, 2.5, 2, 2])
popt, pcov = curve_fit(bleaney_bower, fit_T, fit_chiT, p0=[-10*k, g, 0, 0],  bounds = bounds)
print(popt)
J2 = popt[0]*6.242e18
print(J2)

plt.plot(exp_T, bleaney_bower(exp_T, *popt), color = 'blue', label='Fit') #'Fit: J2=%5.3f, g=%5.3f, C=%5.3f, x=%5.3f' % tuple(popt))
plt.scatter(exp_T, exp_chiT, color = 'r', marker= '.', label='Experiment -- Jiao et al.')
plt.xlabel(r'$T$ [K]')
plt.ylabel(r'$\chi$ $T$ [m$^3$/mol]')
plt.legend()
plt.grid()
#plt.yticks(np.arange(0, 1, 0.1)*1e-5)
plt.xlim(0,300)
plt.show()

J2_dist = np.loadtxt(f"S-T_dist_twist1000_new")
pvals = np.linspace(0,4,1000)
check_dist = J2_dist-J2
bool_dist = np.array([-0.5e-4 < x < 0.5e-4 for x in check_dist])
count = 0
for i in check_dist*bool_dist:
    if i**2 > 1.0e-35:
        print('2J-dist:', J2_dist[count])
        print('U:', pvals[count])
        plt.scatter(pvals[count], J2_dist[count], color = 'r', marker= '.')
    count += 1
#print(check_dist*bool_dist)

#print(J2_dist[81], J2_dist[328], J2_dist[329])
#plt.scatter([pvals[81],pvals[328],pvals[329]],[J2_dist[81], J2_dist[328], J2_dist[329]], color = 'r', marker= '.')
#print(pvals[329], J2_dist[329])

plt.plot(pvals, J2_dist, c='b')
plt.grid()
plt.show()

