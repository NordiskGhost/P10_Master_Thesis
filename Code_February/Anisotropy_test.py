import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

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
freq = 9.241e9              # Microwave frequency in Hz (is fixed due to experimental setup)
gammaL = 16 * 0.175004e-4   # Linewidth (Lorentzian, Tesla)
gammaG = 0.24               # Linewidth (technically the standard deviation, Tesla)
M_S = np.array([1/2, -1/2]) # Possible spin projections of the electrons
S = 1/2

if S == 1:
    Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])/np.sqrt(2)
    Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])/np.sqrt(2)
if S == 1/2:
    Sz = 0.5 * np.array([[1, 0], [0, -1]])
    Sx = 0.5 * np.array([[0, 1], [1, 0]])
    Sy = 0.5 * np.array([[0, -1j], [1j, 0]])

Iz = 0.5 * np.array([[1, 0], [0, -1]])
Ix = 0.5 * np.array([[0, 1], [1, 0]])
Iy = 0.5 * np.array([[0, -1j], [1j, 0]])

Iden2 = np.eye(2)


### fine for spin 1/2, spin 1/2 system

Sx_tot = np.kron(Sx, Iden2) # Makes sure that Sx only works on the electron
Sy_tot = np.kron(Sy, Iden2)
Sz_tot = np.kron(Sz, Iden2)

Ix_tot = np.kron(Iden2, Ix) # Makes sure Ix only works on the nuclei
Iy_tot = np.kron(Iden2, Iy)
Iz_tot = np.kron(Iden2, Iz)

# Now for generalisation
def kron_all(ops): # Does every kronecker products for the allotted operators
    output = ops[0]
    for op in ops[1:]:
        output = np.kron(output, op)
    return output

def electron_op(op, N): # Adds the electron operators everywhere it is appropriate
    return kron_all([op] + [Iden2]*N)   # Iden2 is for the nuclei

def nucleus_op(op, k, N): # Adds nuclei operator for the k'th nuclei
    ops = [Iden2]   # Electron iden2
    for i in range(N):
        ops.append(op if i == k else Iden2) # Nuclei iden2
    return kron_all(ops)

def nuclear_sum(op, N): # Provides the sum: \sum_k(S*I_k)
    return sum(nucleus_op(op, k, N) for k in range(N))

def nuclear_group(op, N, indices): # Provides the sum: \sum_k(S*I_k) for the nuclei within indices, i.e. their equvivalent group
    return sum(nucleus_op(op, k, N) for k in indices)

def lorentzian(x, gamma):
    return gamma / (x**2 + gamma**2)

def gaussian(x, sigma):
    return np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

def iso_H_zeeman(B, E_op, N_op):                                                                                        
    return (g_e * mu_B / h) * B * E_op - (g_N * mu_N / h) * B * N_op

# Euler ZXZ convention
def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0],
                     [np.sin(a),  np.cos(a), 0],
                     [0, 0, 1]])
def Rx(b):
    return np.array([[1, 0, 0],
                     [0, np.cos(b), -np.sin(b)],
                     [0, np.sin(b),  np.cos(b)]])
def rotation_matrix(alpha, beta, gamma = 0):
    return Rz(alpha) @ Rx(beta) @ Rz(gamma)

def basis_index(s_index, n_index):
    return s_index * (2**N) + n_index

def flip_nucleus(state, k):
    return state ^ (1 << k)
def can_flip_up(state, k):
    return ((state >> k) & 1) == 1
def can_flip_down(state, k):
    return ((state >> k) & 1) == 0

def iso_H_hyp(N): # High-field approximation removes x- and y-components.
    global N1, N2, A1, A2, group1, group2
    nuclei_sets = ((N1, A1, group1), (N2, A2, group2))
    H_hyp = np.zeros((2**(N+1), 2**(N+1)))
    E_op = electron_op(Sz, N)
    for i in range(sets):
        H_hyp += nuclei_sets[i][1] * (E_op @ nuclear_group(Iz, N, nuclei_sets[i][2]))
    return H_hyp

def normalizeY(y):
    a, b = y.min(), y.max()
    return (y-a)/(b-a)

def diagonalise_A(A_matrix):
    A_iso = np.trace(A_matrix) / 3
    A_aniso = A_matrix - A_iso * np.eye(3)
    A_aniso_sym = 0.5 * (A_aniso + A_aniso.T)
    eigvals, eigvecs = np.linalg.eigh(A_aniso_sym)
    A_principal = np.diag(eigvals + A_iso)
    return A_principal, eigvals, A_iso, eigvecs

#########################################################

test_arr = np.array([[[-9.08915867e+00, -5.75461306e+00, 0.00000000e+00], [-5.75461306e+00, -1.57340135e+01, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, -1.35471117e+01]],
                     [[-1.86926524e+01, 4.07686664e-15, 0.00000000e+00], [ 4.07686664e-15, -5.65664405e+00, 0.00000000e+00], [ 0.00000000e+00, 0.00000000e+00, -1.32884966e+01]],
                     [[-9.08915867e+00, -5.75461306e+00, 0.00000000e+00], [-5.75461306e+00, -1.57340135e+01, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, -1.35471117e+01]],
                     [[-1.86926524e+01, 4.07686664e-15, 0.00000000e+00], [ 4.07686664e-15, -5.65664405e+00, 0.00000000e+00], [ 0.00000000e+00, 0.00000000e+00, -1.32884966e+01]]])

test_arr = np.array([[[-1.05171343e+01, -6.74892802e+00, 0.00000000e+00], [-6.74892802e+00, -1.83101251e+01, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, -1.58772233e+01]], 
                     [[-2.22174158e+01, 4.87671200e-15, 0.00000000e+00], [4.87671200e-15, -6.62385737e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, -1.58849417e+01]],
                     [[-1.05171343e+01, 6.74892802e+00, 0.00000000e+00], [ 6.74892802e+00, -1.83101251e+01, 0.00000000e+00], [ 0.00000000e+00, 0.00000000e+00, -1.58772233e+01]],
                     [[-2.22174158e+01, 0.00000000e+00, 0.00000000e+00], [0.00000000e+00, -6.62385737e+00, 0.00000000e+00], [0.00000000e+00, 0.00000000e+00, -1.58849417e+01]]])
# Iso values to hard-coded matrix.
#-14.90149424    0
#-14.90873828    1
#-14.90149424    2
#-14.90873828    3
### Iso value test: print(np.trace(test_arr[0])/3)
test_arr *= 1e6


B0 = np.linspace(0.326, 0.333, 2000) #np.linspace(0.336, 0.349, 2000)
freq = 9.241e9  #9.59e9 
spectrum = np.zeros_like(B0)
gammaL = 5.5 * 0.175004e-4        # Linewidth (Lorentzian, Tesla)
gammaG = 4.5 * 0.24e-4
### Multiple nuclei sets
N1, A1 = 2, 12.7901e6
N2, A2 = 2, 12.5459e6
N = N1 + N2
dim = 2**(N+1)
sets = 2


states = np.arange(2**N, dtype=np.uint32) # Gives each state a number
mI = 0.5 - ((states[:, None] >> np.arange(N)) & 1) # Produces a matrix of the nuclei states in binary for each of the states.
# states[:, None] --> produces states as a coloum vector. x >> k shifts our binary bits x by k to the right. E.g. if state 5 = 101, then 5 >> 0 = 5, but 5 >> 1 = 010 = 2
# & 1 --> keeps only 0 (spin up) or 1 (spin down)
mS_vals = np.array([0.5, -0.5])

# Add the hyperfine values at their matrices
A = np.zeros((N, 3, 3))
for i in range(N1):
    A[i] = test_arr[0] # A[i] = test_arr[0],        A[i], _, _, _ = diagnalise_A(test_arr[0])
for j in range(N2):
    A[N1 + j] = test_arr[1] # A[N1 + j] = test_arr[2],       A[N1 + j], _, _, _ = diagnoalise_A(test_arr[2])

for idx in range(len(test_arr)):
    A_p, aniso, iso, vecs = diagonalise_A(test_arr[idx])
    print(f'\nNucleus type {idx}:')
    print(f'  A_iso          = {iso*1e-6:.4f} MHz')
    print(f'  Principal vals = {np.diag(A_p)*1e-6}  MHz')
    print(f'  Anisotropy     = {(aniso.max()-aniso.min())*1e-6:.4f} MHz')


from scipy.sparse import lil_matrix
import itertools
import sys
spinner = itertools.cycle(['-', '/', '|', '\\'])

Sx_full = electron_op(Sx, N)
Sy_full = electron_op(Sy, N)
nr_orientations = 500
phi = np.linspace(0, 2*np.pi, nr_orientations)
theta = np.arccos(np.linspace(-1, 1, nr_orientations))
total_orientations = len(phi) * len(theta)

count = 0
for p_nr, phi_val in enumerate(phi):
    for theta_val in theta:
        H_hf = lil_matrix((dim, dim), dtype=np.float64)
        R = rotation_matrix(phi_val, theta_val)
        Arot = np.zeros_like(A)
        # Sz Iz hyperfine
        for k in range(N):
            Arot[k] = R @ A[k] @ R.T
            A_perp = 0.25 * (Arot[k,0,0] + Arot[k,1,1]) #- 2*Arot[k, 2, 2]
            for ms_i, mS in enumerate(mS_vals):
                for n_state in range(2**N):
                    idx = basis_index(ms_i, n_state)
                    H_hf[idx, idx] += mS * mI[n_state, k] * np.trace(Arot[k])/3   #* Arot[k,2,2]
                    
                #for n_state in range(2**N):
                    if can_flip_down(n_state, k):
                        n_flipped = flip_nucleus(n_state, k)
                        i = basis_index(1, n_state)      # ms = -1/2
                        j = basis_index(0, n_flipped)    # ms = +1/2
                        H_hf[i, j] += A_perp
                        H_hf[j, i] += A_perp
                    if can_flip_up(n_state, k):
                        n_flipped = flip_nucleus(n_state, k)
                        i = basis_index(0, n_state)      # ms = +1/2
                        j = basis_index(1, n_flipped)    # ms = -1/2
                        H_hf[i, j] += A_perp
                        H_hf[j, i] += A_perp
        trans_freq = []
        trans_int  = []
        
        B_ref = B0[len(B0)//2]
        H_ref = H_hf.toarray().copy()
        for ms_i, mS in enumerate(mS_vals):
            for n_state in range(2**N):
                idx = basis_index(ms_i, n_state)
                H_ref[idx, idx] += (g_e * mu_B / h) * B_ref * mS
                H_ref[idx, idx] -= (g_N * mu_N / h) * B_ref * np.sum(mI[n_state])
        
        evals_ref, evecs_ref = np.linalg.eigh(H_ref)
        
        for i in range(dim):
            for j in range(i+1, dim):
                intensity = (abs(evecs_ref[:, i].conj() @ Sx_full @ evecs_ref[:, j])**2
                           + abs(evecs_ref[:, i].conj() @ Sy_full @ evecs_ref[:, j])**2)
                if intensity > 1e-20:
                    trans_freq.append(evals_ref[j] - evals_ref[i])
                    trans_int.append(intensity)
        
        trans_freq = np.array(trans_freq)
        trans_int  = np.array(trans_int)
        # For allowed EPR transitions delta_mS = 1
        freq_shift = (g_e * mu_B / h) * (B0 - B_ref)
        dfreq_grid = trans_freq[None, :] + freq_shift[:, None]
        
        lor = lorentzian(dfreq_grid - freq, gammaL * (g_e * mu_B / h))
        gauss = gaussian(dfreq_grid - freq, gammaG * (g_e * mu_B / h))
        spectrum += (lor * trans_int[None, :]).sum(axis=1)
        
        #sys.stdout.write(next(spinner))  # write the next character
        #sys.stdout.flush()               # flush stdout buffer (actual character display)
        #sys.stdout.write('\b')           # erase the last written char
        
        count += 1
        print(f'Percent done: {round(100 * (count + 1) / total_orientations, 3)} \r', end ='')

spectrum /= total_orientations
deriv_spectrum = normalizeY(np.gradient(spectrum, B0[1] - B0[0])) #* 0.8 + 0.1

df = pd.read_excel('Fig5_data.xlsx', sheet_name='5c', usecols='C:D')
#df = pd.read_excel('Fig4-plots.xlsx', sheet_name='4c', usecols='N:O') #'C:D' for exp, 'N:O' for simulated spectrum
data = df.to_numpy()
data[:, 1] = normalizeY(data[:, 1])

print('Lorentz gamma [MHz]:', round(gammaL * (g_e * mu_B / h)*1e-6, 3))
plt.plot(B0*1e3, deriv_spectrum, color='blue', label='Simulated')
plt.plot(data[:, 0], data[:, 1], color='black', label='Jiao et al.')
plt.xlabel("Magnetic Field [mT]")
plt.legend()
plt.ylabel("Intensity [a.u.]")
plt.grid()
plt.xlim(min(B0*1e3), max(B0*1e3))
plt.show()


print("mI shape:", mI.shape)
print("Expected:", (2**N, N))
print("First few rows:\n", mI[:8])

trans_freq = np.array(trans_freq)
trans_int = np.array(trans_int)
print("Number of transitions:", len(trans_freq))
print("Max intensity:", trans_int.max())
print("Min intensity:", trans_int.min())
print("Intensity distribution:", np.sort(trans_int)[::-1][:10])