import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import time
start_time = time.time()
# Continuous wave EPR (CW-EPR) MVP
######################################################################
###                        Isotropic case                          ###
######################################################################

# Constants
# X, Q, W, S bands for g = 2
# X-band: 9-10 GHz (B0 0.32-0.36 T), Q-band: 33-35 GHz (B0 1.2-1.3 T)
# W-band: 95 GHz (B0 3.4 T), S-band: 2-4 GHz (B0 0.07-0.42 T)
g_e = 2.00231930436082      # g-factor for free electron
mu_B = 9.274e-24            # Bohr magneton (J/T)
mu_N = 5.05e-27             # Nuclear magneton (J/T) # FOR nuclear Zeeman
g_N = 5.585                 # Proton g-factor        # FOR nuclear Zeeman
h = 6.626e-34               # Planck constant (J*s)
hbar = h/(2*np.pi)          # Reduced Planck constant

# Simulation parameters
B0 = np.linspace(0.336, 0.349, 4000)  # Magnetic field sweep in Tesla
freq = 9.59e9                # Microwave frequency in Hz (is fixed due to experimental setup)
gammaL = 1.85*0.175004e-3        # Linewidth (Lorentzian, Tesla)
#gammaG = 0.24e-3               # Linewidth (technically the standard deviation, Tesla)

def lorentzian(x, gamma):
    return gamma / (x**2 + gamma**2)

def energies(B, mI, H_hyp_diag, mS_up, mS_down):
    z_e = (g_e * mu_B / h) * B # electron zeeman
    z_n = (g_N * mu_N / h) * B # nuclear zeeman
    E_up   =  z_e * mS_up   - z_n * np.sum(mI, axis=1) + H_hyp_diag * mS_up # spin up block (highest energy)
    E_down =  z_e * mS_down - z_n * np.sum(mI, axis=1) + H_hyp_diag * mS_down # spin down block (lowest energy)
    return E_up, E_down

def CWEPR(N1, A1, N2, A2):
    global B0
    ### This code can be used later for anisotropic effects! And it works, however, it struggles with too large matricies --> too much memory allocation
    # The following code only works for CW-EPR and the high-field approximation! It makes the Hamiltonian a vector since it is only the diagonal elements
    # which contribute to the energies.
    spectrum = np.zeros_like(B0)
    #B0 = np.linspace(0.336, 0.349, 4000)
    ### Multiple nuclei sets
    #N1, A1 = 4, 18.09079919e6
    #N2, A2 = 4, 18.18233706e6
    #N3, A3 = 0, 0
    #N4, A4 = 0, 0
    N = N1 + N2 #+ N3 + N4
    dim_N = 2**N

    states = np.arange(dim_N, dtype=np.uint32) # Gives each state a number
    mI = 0.5 - ((states[:, None] >> np.arange(N)) & 1) # Produces a matrix of the nuclei states in binary for each of the states.
    # states[:, None] --> produces states as a coloum vector. x >> k shifts our binary bits x by k to the right. E.g. if state 5 = 101, then 5 >> 0 = 5, but 5 >> 1 = 010 = 2
    # & 1 --> keeps only 0 (spin up) or 1 (spin down)

    mS_up = 0.5
    mS_down = -0.5
    # Add the hyperfine values at their appropriate "diagonals" in the vector
    A = np.zeros(N)
    A[:N1] = A1
    A[N1:N1+N2] = A2
    #A[N1+N2:N1+N2+N3] = A3
    #A[N1+N2+N3:] = A4
    H_hyp_diag = np.dot(mI, A) # shape --> (2^N, ) (a vector of "diagonals")

    percentage = len(B0)
    for i, B in enumerate(B0):
        E_up, E_down = energies(B, mI, H_hyp_diag, mS_up, mS_down)
        dfreq = E_up - E_down
        spectrum[i] = np.sum(lorentzian(dfreq - freq, gammaL * (g_e * mu_B / h)))
        print(f'Percent done: {round(100 * (i + 1) / percentage, 3)} \r', end ='')
    deriv_spectrum = np.gradient(spectrum, B0[1] - B0[0])
    return deriv_spectrum

### Neutral
B0 = np.linspace(0.336, 0.349, 4000)
freq = 9.59e9
gammaL = 1.45*0.175004e-3
deriv_spectrum = CWEPR(2, (12.54593099)*1e6, 2, (12.7900946)*1e6)
#deriv_spectrum2 = CWEPR(2, 12.1955e6, 2, 12.6311e6)


# Percent-wise derivation
print((12.54593099-12.1955)/(12.1955), (12.7900946-12.6311)/(12.6311))

df = pd.read_excel('Fig4-plots.xlsx', sheet_name='4c', usecols='C:D')
df2 = pd.read_excel('Fig4-plots.xlsx', sheet_name='4c', usecols='N:O')
data = df.to_numpy()
data2 = df2.to_numpy()

max_idx = argrelextrema(deriv_spectrum, np.greater)[0]
min_idx = argrelextrema(deriv_spectrum, np.less)[0]
peak_peak = np.zeros(len(max_idx))
for i in range(len(max_idx)):
    dist = B0[min_idx[i]]*1e3 - B0[max_idx[i]]*1e3
    peak_peak[i] = dist
    plt.scatter(B0[max_idx[i]]*1e3, deriv_spectrum[max_idx[i]] * 12, color='black', marker='.')
    plt.scatter(B0[min_idx[i]]*1e3, deriv_spectrum[min_idx[i]] * 12, color='black', marker='.')
    T2_star = h / (g_e * mu_B * np.pi * np.sqrt(3) * dist*1e-3)  # seconds  "2/(np.sqrt(3) * gamma_e * dist*1e-3)" does the same when gamma_e = 1.76085962784 * 10**11
    print('No saturation:', T2_star * 1e6)  # microseconds

print(peak_peak) # mT
print('T2* Gaussian:', 2*np.sqrt(2)*hbar/(g_e * mu_B * peak_peak*1e-3))
print('Testing test', (g_e * mu_B)/(np.pi* hbar *peak_peak*1e-3))
print('T2* Gaussian:', 2*np.sqrt(2)*hbar/(g_e * mu_B * 0.1574*1e-3))
print('T2* Gaussian:', 2*np.sqrt(2)*hbar/(g_e * mu_B * 0.1566*1e-3))
print(r'T1 natural linewidth [$\mu$s]:', np.average(1/peak_peak))
print(r'T2 natural linewidth [$\mu$s]:', 0.5* np.average(0.318/peak_peak))
print('Test T2', 1.3131e-7/(g_e*(peak_peak)*10))
print('Test T1', (0.9848e-7*(np.average(peak_peak)*10)/g_e)*(250))




#### Testing previous data
print('='*50)
max_idx = argrelextrema(data2[:, 1]/355, np.greater)[0]
min_idx = argrelextrema(data2[:, 1]/355, np.less)[0]
peak_peak = np.zeros(len(max_idx))
for i in range(len(max_idx)):
    dist = data2[min_idx[i]][0] - data2[max_idx[i]][0]
    peak_peak[i] = dist
    plt.scatter(data2[min_idx[i], 0], data2[min_idx[i], 1]/355, color='black', marker='.')
    plt.scatter(data2[max_idx[i], 0], data2[max_idx[i], 1]/355, color='black', marker='.')
    T2_star = h / (g_e * mu_B * np.pi * np.sqrt(3) * dist*1e-3)  # seconds  "2/(np.sqrt(3) * gamma_e * dist*1e-3)" does the same when gamma_e = 1.76085962784 * 10**11
    print('No saturation:', T2_star * 1e6)  # microseconds

print(peak_peak) # mT
print('T2* Gaussian:', 2*np.sqrt(2)*hbar/(g_e * mu_B * peak_peak*1e-3))
print('T2* Lorentz:', h / (g_e * mu_B * np.pi * np.sqrt(3) * peak_peak*1e-3))
print(r'T1 natural linewidth [$\mu$s]:', np.average(1/peak_peak))
print(r'T2 natural linewidth [$\mu$s]:', 0.5* np.average(0.318/peak_peak))
print('Test T2', 1.3131e-7/(g_e*(peak_peak)*10))
print('Test T1', (0.9848e-7*(np.average(peak_peak)*10)/g_e)*(250))





    
print('Lorentz gamma [MHz]:', round(gammaL * (g_e * mu_B / h)*1e-6, 3))
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(B0*1e3, deriv_spectrum * 12, color='blue', label='Simulated')
#plt.plot(data[:, 0], data[:, 1]/355, color='black', label='Jiao et al.')
plt.plot(data2[:, 0], data2[:, 1]/355, color='red', label='Jiao et al. Simulation')
plt.legend()
plt.xlabel("Magnetic Field [mT]")
plt.ylabel("Intensity [a.u.]")
plt.grid()
plt.xlim(min(B0*1e3), max(B0*1e3))
plt.show()

def normalizeY(y):
    a, b = y.min(), y.max()
    return (y-a)/(b-a)


df = pd.read_excel('Fig5_data.xlsx', sheet_name='5c', usecols='C:D')
data = df.to_numpy()
data[:, 1] = normalizeY(data[:, 1])

### Cation
B0 = np.linspace(0.326, 0.334, 4000)
freq = 9.240e9 
gammaL = 10*0.175004e-4 
deriv_spectrum = (1e3/9)*CWEPR(2, 14.90149424e6, 2, 14.90873828e6) + 0.5#CWEPR(2, 17.001e6, 2, 16.8e6))
deriv_spectrum2 = (1e3/6.6)*CWEPR(2, 11.1245e6, 2, 14.5349e6) + 0.5


print('Lorentz gamma [MHz]:', round(gammaL * (g_e * mu_B / h)*1e-6, 3))
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(B0*1e3, deriv_spectrum, color='blue', label='Simulated')
plt.plot(B0*1e3, deriv_spectrum2, color='red', label='Simulated')
plt.plot(data[:, 0], data[:, 1], color='black', label='Jiao et al.')
plt.xlabel("Magnetic Field [mT]")
plt.legend()
plt.ylabel("Intensity [a.u.]")
plt.grid()
plt.xlim(min(B0*1e3), max(B0*1e3))
plt.clf()


"""

### Naphthalene
print((18.84740672 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (4.660251 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.76e-4 * (g_e * mu_B / h)*1e-6))
print((15.36366209 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (4.95547196 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.76e-4 * (g_e * mu_B / h)*1e-6))
print((15.07088417 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (5.10883427 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.76e-4 * (g_e * mu_B / h)*1e-6))
print((17.37773478 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (3.88882962 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.76e-4 * (g_e * mu_B / h)*1e-6))


def plot_exp_naphthalene(B_offset):
    df = pd.read_csv('plot-data.csv')
    data = df.to_numpy()
    x, y = data[:, 0], data[:, 1]
    y /= y.max()
    plt.plot(B_offset + (x[:-1] - x[-1]), y[:-1], color = 'b', label = 'Weil & Bolten')
    #plt.show()

gammaL = 0.05e-4        # Linewidth (Lorentzian, Tesla)
B0 = np.linspace(0.336, 0.349, 4000)
freq = 9.59e9 
deriv_spectrum = CWEPR(4, 15.07088417e6, 4, 5.10883427e6) # U = 1.6
#deriv_spectrum2 = CWEPR(4, 0.495e-3 * (g_e * mu_B / h), 4, 0.187e-3 * (g_e * mu_B / h)) # Weil & Boltons values, but not the actual plot
deriv_spectrum2 = CWEPR(4, 18.84740672e6, 4, 4.660251e6) # U = 4.0
print(0.495e-3 * (g_e * mu_B / h), 0.187e-3 * (g_e * mu_B / h))

deriv_spectrum /= deriv_spectrum.max()
deriv_spectrum2 /= deriv_spectrum2.max()

plot_exp_naphthalene(1e3*B0[np.argmin(deriv_spectrum)])


print('Lorentz gamma [MHz]:', round(gammaL * (g_e * mu_B / h)*1e-6, 3))
print("--- %s seconds ---" % (time.time() - start_time))
#plt.plot(B0*1e3, deriv_spectrum, color='r', label = 'Simulated')
plt.plot(B0*1e3, deriv_spectrum2, color='r', label = 'Simulated U = 4.0')
plt.plot(B0*1e3, deriv_spectrum, color='g', label = 'Simulated U = 1.6')
plt.xlabel("Magnetic Field [mT]")
plt.ylabel("Intensity [a.u.]")
plt.legend()
plt.grid()
plt.xlim(340.56, 343.8)
plt.show()




gammaL = 0.17004e-4        # Linewidth (Lorentzian, Tesla)
B0 = np.linspace(0.336, 0.349, 4000)
freq = 9.59e9 
deriv_spectrum = CWEPR(2, 25e6, 2, 7e6)

deriv_spectrum /= deriv_spectrum.max()

print('Lorentz gamma [MHz]:', round(gammaL * (g_e * mu_B / h)*1e-6, 3))
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(B0*1e3, deriv_spectrum, color='r', label = 'Simulated')
plt.xlabel("Magnetic Field [mT]")
plt.ylabel("Intensity [a.u.]")
plt.legend()
plt.grid()
plt.show()

"""