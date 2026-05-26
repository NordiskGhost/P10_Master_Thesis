import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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
B0 = np.linspace(0.330, 0.350, 12000)  # Magnetic field sweep in Tesla
freq = 9.5e9                # Microwave frequency in Hz (is fixed due to experimental setup)
gammaL = 0.04e-3        # Linewidth (Lorentzian, Tesla)
gammaG = 0.24*0         # Linewidth (technically the standard deviation, Tesla)
M_S = np.array([1/2, -1/2]) # Possible spin projections of the electrons

# Lorentzian line shape     #https://mathworld.wolfram.com/LorentzianFunction.html
def lorentzian(x, x0, width):
    return (width**2) / (np.pi*((x - x0)**2 + width**2))

def pascal(n):
    if n==0:
        return [1]
    else:
        N = pascal(n-1)
        return np.array([1] + [N[i] + N[i+1] for i in range(n-1)] + [1])

# We need to determine when the resonance occurs, so we define a "resonance field"
# Moreover, we need the intensites at these resonances so we define "intesities"
resonance_fields = []
combined_intensities = []

######## Equivalent nuclei
# Number of equiv. nuclei with I = 1/2
n_tot = 2 # number of different nuclei with different equiv. nuclei
          # (e.g. CH2OCH3 has 2 equiv. nuclei in CH2 and 3 equiv. nuclei in CH3)
n1 = 4
n2 = 8
n3 = 2
n4 = 4
# 4, 8, 2, 4 ############################### Probably the true distribution.


I_tot_1 = n1/2
I_tot_2 = n2/2
I_tot_3 = n3/2
I_tot_4 = n4/2

M_I_1 = np.linspace(-n1/2, n1/2 + 1, n1 + 1) # Possible spin projections of the nuclei (assuming 13C???)
M_I_2 = np.linspace(-n2/2, n2/2 + 1, n2 + 1)
M_I_3 = np.linspace(-n3/2, n3/2 + 1, n3 + 1)
M_I_4 = np.linspace(-n4/2, n4/2 + 1, n4 + 1)

intensities_1 = pascal(n1)
intensities_2 = pascal(n2)
intensities_3 = pascal(n3)
intensities_4 = pascal(n4)
intensities_1 = intensities_1 / np.sum(intensities_1)
intensities_2 = intensities_2 / np.sum(intensities_2)
intensities_3 = intensities_3 / np.sum(intensities_3)
intensities_4 = intensities_4 / np.sum(intensities_4)

######## Hyperfine constants
# List of used hyperfine constants:
A_1 = 4.89e-4                     # Hyperfine coupling in Tesla (1 G = 0.1 mT)  # In the code we use it
A_2 = 1.81e-4
A_3 = 1.81e-4
A_4 = 1.81e-4
# in A * m_S * h, where we need to convert it into Hz, so:
A1_Hz = A_1 * g_e * mu_B / h   # Convert to Hz
A2_Hz = A_2 * g_e * mu_B / h
A3_Hz = A_3 * g_e * mu_B / h
A4_Hz = A_4 * g_e * mu_B / h

#### If given in MHz
A1_Hz, A2_Hz, A3_Hz, A4_Hz = 12.1955e6, 12.6311e6, 10.9772e6, 13.9595e6
####


# We have transition rule Delta M_S = +-1 and Delta M_I = 0, so we only "loop" over M_S values.
# In reality we loop over the M_I values since the energy difference is independent of M_S
#for m_I in M_I:
    # The energies can be described by ge*μB*B*M_S - gN*μN*B*M_I + h*A*M_I*M_S
    # The energy difference when using the transition rule is then going to be:
    # ge*μB*B +- 1/2 * h*A, where (+) is for M_I = -1/2 and (-) for M_I = 1/2
    # We then solve for for B_res in h*ν = gμB*B + A*m_I
#    B_res = (h * freq - A_Hz * m_I * h) / (g_e * mu_B)
#    resonance_fields.append(B_res)

count1 = 0
for m1 in M_I_1:
    count2 = 0
    for m2 in M_I_2:
        count3 = 0
        for m3 in M_I_3:
            count4 = 0
            for m4 in M_I_4:
                total_mI = A1_Hz * m1 + A2_Hz * m2 + A3_Hz * m3 + A4_Hz * m4
                B_res = (h * freq - total_mI * h) / (g_e * mu_B)
                intensity = intensities_1[count1] * intensities_2[count2]  * intensities_3[count3]  * intensities_4[count4]
                resonance_fields.append(B_res)
                combined_intensities.append(intensity)
                count4 += 1
            count3 += 1
        count2 += 1
    count1 += 1

# Generate spectrum
spectrum = np.zeros_like(B0)
for B_res, intensity in zip(resonance_fields, combined_intensities):
    spectrum += intensity * lorentzian(B0, B_res, gammaL)
#test = gaussian_filter(spectrum, sigma = gammaG)

# When measuring experimentally the first derivative dL/dB (where L is the lorentzian line)
# is usually used. Thus we find the numerical derivative of the spectrum w.r.t. B0. B0[1] - B0[0]
# which subsequently determines our step-size.
deriv_spectrum = np.gradient(spectrum, B0[1] - B0[0])


# Plot
#plt.plot(B0 * 1e3, spectrum/np.max(np.abs(spectrum)), c='r', label='0th derivative')
plt.plot(B0 * 1e3, deriv_spectrum / np.max(np.abs(deriv_spectrum)), c='b', label='1st derivative', linewidth = 0.7)
plt.xlabel("Magnetic Field [mT]")
plt.ylabel("Intensity [a.u.]")
plt.legend()
#plt.title("Simulated CW-EPR Spectrum (S=1/2, I=1/2)")
plt.xlim(334,342.5)
plt.ylim(-1,1)
plt.grid()
plt.show()


