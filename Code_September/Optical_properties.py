import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants, h, k
from collections import defaultdict

# Physical constants
g_e = physical_constants['electron g factor'][0]
mu_B = physical_constants['Bohr magneton'][0]       # J/T
mu_N = physical_constants['nuclear magneton'][0]    # J/T


CC_BOND = 1.42
CC_CUTOFF = 1.6
CH_BOND = 1.09


#####  Structure
xyz = []
### Naphthatalene
for j in range(6):
    theta = j * np.pi / 3 + (np.pi/6) # The 30deg is to turn the hexagons on its side
    xyz.append([np.cos(theta), np.sin(theta), 0])
    if j in [2,3]:
        pass
    else:
        xyz.append([2*np.sqrt(3)/2 + np.cos(theta), np.sin(theta), 0])
xyz = np.array(xyz) * CC_BOND
###

### Clar's goblet
def generateTriangulene():
    xyz = []
    ### First layer
    for j in range(6):
        theta = j * np.pi / 3 + (np.pi/6) # The 30deg is to turn the hexagons on its side
        xyz.append([np.cos(theta), np.sin(theta)-2.5, 0])
    for k in range(2,5,2):
        for i in [0, 1, 4, 5]:
            theta = i * np.pi / 3 + (np.pi/6)
            xyz.append([np.cos(theta)+k*np.sqrt(3)/2, np.sin(theta)-2.5, 0])
    ### Second layer
    for j in [0,1,2]:
        theta = j * np.pi / 3 + (np.pi/6) # The 30deg is to turn the hexagons on its side
        xyz.append([np.cos(theta)+2*np.sqrt(3)/4, np.sin(theta)+1.5-2.5, 0])
    ### Third layer
    for i in [0, 1]:
        theta = i * np.pi / 3 + (np.pi/6)
        xyz.append([np.cos(theta)+3*np.sqrt(3)/2, np.sin(theta)+1.5-2.5, 0])
    return np.array(xyz)

def generateGoblet():
    ### Turn and duplicate triangulene structure to form Clar's goblet
    lower = generateTriangulene()
    upper = lower.copy()
    upper[:, 1] = -upper[:, 1]+1
    whole = np.concatenate((lower, upper), axis = 0)
    return whole * CC_BOND

def generateTwistedGoblet():
    ### Turn and duplicate triangulene structure to form Clar's goblet
    lower = generateTriangulene()
    upper = lower.copy()
    upper[:, 1] = -upper[:, 1]+1
    twist_lower = np.zeros((len(lower), 4))
    twist_upper = np.zeros((len(lower), 4))
    count = 0
    for i, j in zip(lower, upper):
        twist_lower[count] = np.append(i * CC_BOND, 1)
        twist_upper[count] = np.append(j * CC_BOND, 0)
        if i[1] > -0.01:
            twist_lower[count] = np.append(i * CC_BOND, 1)
        count += 1
    whole = np.concatenate((twist_lower, twist_upper), axis = 0)
    return whole

def add_hydrogen(xyz):
    xyz = np.asarray(xyz)
    nC = len(xyz)
    neighbors = [[] for _ in range(nC)]
    for i in range(nC):
        for j in range(i+1, nC):
            d = np.linalg.norm(xyz[i] - xyz[j])
            if d < CC_CUTOFF:
                neighbors[i].append(j)
                neighbors[j].append(i)
    hydrogens = []
    for i, nbrs in enumerate(neighbors):
        coord = len(nbrs)
        # sp2 carbon wants 3 bonds
        missing = 3 - coord
        if missing <= 0:
            continue
        ri = xyz[i]
        # Sum of bond vectors to neighbors
        bond_vecs = np.zeros(3)
        for j in nbrs:
            bond_vecs += (xyz[j] - ri)
        # Outward direction is opposite of bond sum
        if np.linalg.norm(bond_vecs) < 1e-6:
            continue
        direction = -bond_vecs / np.linalg.norm(bond_vecs)
        # Add hydrogens (usually 1 for triangulene edges)
        for _ in range(missing):
            h_pos = ri + CH_BOND * direction
            hydrogens.append(h_pos)
    return np.array(hydrogens)
###
xyz = generateGoblet()
twist = True
if twist == False:
    pass
else:
    xyz_twist = generateTwistedGoblet()
    for nr, i in enumerate(xyz_twist[:,3:]):
        plt.annotate(f'{i[0]}', (xyz[:, 0][nr], xyz[:, 1][nr]))
plt.grid()
plt.scatter(xyz[:, 0], xyz[:, 1])
plt.axis('equal')
plt.clf()
#####



#####  Energy and spin densities
def mult_sort(arr, tolerance = 0.3, print_it = False):
    if isinstance(arr, np.ndarray):
            arr = arr.tolist()

    # Group similar values using bins
    grouped = defaultdict(list)
    for value in arr:
        # Only works for numeric values
        placed = False
        for key in grouped:
            if abs(value - key) <= tolerance:
                grouped[key].append(value)
                placed = True
                break
        if not placed:
            grouped[value].append(value)

    # Sort and summarize
    sorted_items = sorted(
        [(round(np.mean(values), 3), len(values)) for values in grouped.values()]
    )
    # Display
    if print_it == True:
        print("Value\tMultiplicity")
        for value, multiplicity in sorted_items:
            print(f"{value}\t{multiplicity}")
    return sorted_items

def theta_kron(a, b, theta):
    if int(a[0]) == int(b[0]):
        return 1
    else:
        return np.cos(np.deg2rad(theta))**2

def Hubbard(U, S_z):
    # Hubbard parameter U [eV]
    global xyz, twist
    delta1 = 2.7
    delta2 = 0.20
    delta3 = 0.18
    
    theta = 9.6 #37 # degrees
    # Spinless Hamiltonian H0
    N = len(xyz) # Num. sites and electrons (at half-filling)  
    H0 = np.zeros((N, N))
    if twist == True:
        twist_matrix = np.zeros((N, N))
        for a in range(N):
            for b in range(N):
                twist_matrix[a,b] = theta_kron(xyz_twist[:,3:][a], xyz_twist[:,3:][b], theta)
    else:
        twist_matrix = np.ones((N,N))
    ##################################################################
    ### FOR THE SUPPLEMENTARY INFORMATION OF NATURE ARTICLE: https://www.nature.com/articles/s41557-025-01776-1
    first = np.zeros((N,N), dtype=int)
    second = np.zeros((N,N), dtype=int)
    third = np.zeros((N,N), dtype=int)
    distances = np.linalg.norm(xyz[:, np.newaxis] - xyz, axis=2)
    for i in range(N):
        nearest = mult_sort(distances[i])[1:4]
        nearest = [i[0] for i in nearest]
        first[i] = np.isclose(distances[i], nearest[0], atol=0.1)
        second[i] = np.isclose(distances[i], nearest[1], atol=0.1)
        third[i] = np.isclose(distances[i], nearest[2], atol=0.1)
        #first[i] = np.isclose(distances[i], 1.42, atol=0.1) #1.00 1.42
        #second[i] = np.isclose(distances[i], 2.46, atol=0.1) #1.73 2.46
        #third[i] = np.isclose(distances[i], 2.0, atol=0.1) #2.646 2.84
    H0 += -delta1*first*twist_matrix-delta2*second-delta3*third
    ##################################################################
    """
    ### FOR THE RIGID AND REGULAR STRUCTURE USING GEN-FUNCTIONS
    # Spinless Hamiltonian H0 (i.e. all but the diagonal)
    H0 = np.zeros((N, N))
    distances = np.linalg.norm(xyz[:, np.newaxis] - xyz, axis=2)    
    H0[np.isclose(distances, 1, atol=0.1)] = -delta1
    H0[np.isclose(distances, 1.73, atol=0.1)] = -delta2
    H0[np.isclose(distances, 2, atol=0.1)] = -delta3
    """
    ##################################################################
    
    ### Spin part
    # Double the system size to accomodate for spin, i.e. two spin blocks: spin-up [0:N], spin-down [N:2N]
    H_spin = np.zeros((2*N, 2*N))
    H_spin[:N, :N] = H0  # Spin-up block
    H_spin[N:, N:] = H0  # Spin-down block

    # Use mean-field approximation for the Hubbard parameter U.
    # This uses an initial guess for the distribution of spin up and spin down electrons in the structure
    # Later we iterate in a "self-consistent" loop to achieve occupation values for n_up and n_down, that
    # does not change. This happens by calculating the the eigenvalues and -vectors of H_total, occupying
    # half of the allowed states (half-filling) and noting if n_up and n_down have significantly changed
    # from the previous iteration.
    

    num_electrons = N # Assume one electron per site (half-filling)
    # Occupy the spin up and spin down separately
    num_up = round(num_electrons/2 + S_z) # Number of spin up
    num_down = round(num_electrons/2 - S_z) # Number of spin down

    rng = np.random.default_rng(1701)
    
    n_up = np.ones(N) * (num_up)/N * rng.normal(0, 1, size = N)            #alt. to "* 1.3" 
    n_down = np.ones(N) * (num_down)/N * rng.normal(0, 1, size = N)        #alt. to "* 0.7"
    alpha = 0.2

    # The self-consistent loop
    for iteration in range(10000):
        #H_off_diag = np.zeros((N,N))
        H_U = np.zeros((2*N, 2*N)) # The on-site Hubbard terms
        
        # Adding the Hubbard parameter to each of the spin-up and spin-down states
        H_U[:N, :N] += np.diag(U * (n_down-1/2))  # for spin-up
        H_U[N:, N:] += np.diag(U * (n_up-1/2))    # for spin-down
        
        H_total = H_spin + H_U

        H_up = H_total[:N, :N] # Hamilton spin up
        H_down = H_total[N:, N:] # Hamilton spin down
        
        eigvals_up, eigvecs_up = np.linalg.eigh(H_up)
        eigvals_down, eigvecs_down = np.linalg.eigh(H_down)

        # Since we earlier did split up the Hamiltonian in spin up for [0:N] and spin down for [N:2N] parts
        # we use the pairs for each. So H_up for spin up and H_down for spin down.
        idx = np.argsort(eigvals_up)
        occ_states_up = eigvecs_up[:, idx[:num_up]]
        idx = np.argsort(eigvals_down)
        occ_states_down = eigvecs_down[:, idx[:num_down]]
        
        #occ_states_up = eigvecs_up[:, :num_up]
        #occ_states_down = eigvecs_down[:, :num_down]
        
        # Project onto site occupations
        n_up_new = np.sum(np.abs(occ_states_up)**2, axis=1)
        n_down_new = np.sum(np.abs(occ_states_down)**2, axis=1)
        
        # Checking the convergence of n_up and n_down
        tolerance = 1e-12
        if np.max(np.abs(n_up - n_up_new)) < tolerance and np.max(np.abs(n_down - n_down_new)) < tolerance:
            print(f"Converged after {iteration} iterations.")
            #print(H_total) # Sanity check
            break
        elif iteration == 9999:
            print("Max iteration, did not converge")
        # Linear comb. for faster convergence. Uses the electron densities from both new and old n.
        n_up = alpha * n_up_new + (1 - alpha) * n_up
        n_down = alpha * n_down_new + (1 - alpha) * n_down
    
    E_kinetic = np.sum(eigvals_up[:num_up]) + np.sum(eigvals_down[:num_down])
    E_interaction = -U * np.sum((n_up_new-1/2) * (n_down_new-1/2))
    total_energy = E_kinetic + E_interaction/2
    #print("tot:", total_energy)
    #print("kin:", E_kinetic)
    #print("U:", E_interaction)
    #print("N:", N)
    print(n_up_new)
    print(n_down_new)
    spin_den = n_up_new - n_down_new
    return H_total, total_energy, spin_den

def plot_polarisability(U, S_z):
    H_total, _, spin_den = Hubbard(U, S_z)
    N = len(spin_den)
    eigenvalues, eigenvectors = np.linalg.eigh(H_total)
    occupied = len(eigenvectors) // 2
    c_n = eigenvectors[:, :occupied]
    c_m = eigenvectors[:, -occupied:]
    
    x = xyz[:, 0]
    X_op = np.zeros((2*N, 2*N))
    X_op[:N, :N] = np.diag(x)
    X_op[N:, N:] = np.diag(x)
    y = xyz[:, 1]
    Y_op = np.zeros((2*N, 2*N))
    Y_op[:N, :N] = np.diag(y)
    Y_op[N:, N:] = np.diag(y)
    z = xyz[:, 1]
    Z_op = np.zeros((2*N, 2*N))
    Z_op[:N, :N] = np.diag(z)
    Z_op[N:, N:] = np.diag(z)
    
    braketx = c_n.conjugate().T @ X_op @ c_m # |<n|x|m>|
    brakety = c_n.conjugate().T @ Y_op @ c_m # |<n|y|m>|
    braketz = c_n.conjugate().T @ Z_op @ c_m # |<n|z|m>|
    
    hbar = 1
    e = 1
    
    def alpha_xx(omega, braket):
        E_nm = eigenvalues[occupied:] - eigenvalues[:occupied, np.newaxis]
        braket_squared = braket**2
        omega_squared = (omega + 1j * 0.05) * hbar
        return 2 * e**2 * np.sum(braket_squared * E_nm / (E_nm**2 - omega_squared**2))
    
    omega_values = np.linspace(1, 5, 5000) # 1/s
    alpha_Xvalues = np.array([alpha_xx(omega, braketx) for omega in omega_values])
    alpha_Yvalues = np.array([alpha_xx(omega, brakety) for omega in omega_values])
    alpha_Zvalues = np.array([alpha_xx(omega, braketz) for omega in omega_values])

    alpha_values = (alpha_Xvalues + alpha_Yvalues + alpha_Zvalues)/3
    
    #plt.subplot(1, 2, 1)
    #plt.plot(omega_values, np.imag(alpha_values), c='r')
    #plt.plot(omega_values, np.real(alpha_values), c='b')
    #plt.legend(["Im", "Re"])
    #plt.grid()
    #plt.xlabel(r'$\hbar\omega$ [eV]')
    #plt.ylabel(r'$\alpha$')

    #plt.subplot(1, 2, 2)
    hbar_SI = 1.054571800e-34 # J*s
    c = 299792458e9 # nm/s
    wavelength = c/((hbar*omega_values*1.60218e-19)/(2*np.pi*hbar_SI)) #1.60218e-19 convert eV -> J
    plt.plot(wavelength, np.imag(alpha_values), c='r', label = 'Im - Neutral')
    plt.plot(wavelength, np.real(alpha_values), c='b', label = 'Re - Neutral')
    plt.legend()
    plt.grid()
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'$\alpha$ [A$^2$ s$^4$ kg$^{-1}$]')
    plt.xlim(250,800)
    
    #plt.clf()
    plt.show()

def plot_polarisability_test(U, S_z):
    H_total, _, spin_den, num_up, num_down = Hubbard(U, S_z)
    N = len(spin_den)
    
    # Extract spin-resolved Hamiltonians from H_total
    H_up = H_total[:N, :N]
    H_down = H_total[N:, N:]
    
    # Diagonalise each spin block separately
    eigvals_up, eigvecs_up = np.linalg.eigh(H_up)
    eigvals_down, eigvecs_down = np.linalg.eigh(H_down)
    
    # Separate occupied and unoccupied states per spin channel
    # (this naturally enforces the spin selection rule Δm_s = 0)
    c_n_up   = eigvecs_up[:, :num_up]      # occupied spin-up
    c_m_up   = eigvecs_up[:, num_up:]      # unoccupied spin-up
    c_n_down = eigvecs_down[:, :num_down]  # occupied spin-down
    c_m_down = eigvecs_down[:, num_down:]  # unoccupied spin-down
    
    x = xyz[:, 0]
    y = xyz[:, 1]
    
    # Compute dipole matrix elements within each spin channel only
    braketx_up   = c_n_up.conj().T   @ np.diag(x) @ c_m_up
    brakety_up   = c_n_up.conj().T   @ np.diag(y) @ c_m_up
    braketx_down = c_n_down.conj().T @ np.diag(x) @ c_m_down
    brakety_down = c_n_down.conj().T @ np.diag(y) @ c_m_down

    hbar = 1
    e = 1

    def alpha_xx(omega, braket, E_occ, E_unocc):
        # E_nm shape: (num_occ, num_unocc) via broadcasting
        E_nm = E_unocc[np.newaxis, :] - E_occ[:, np.newaxis]
        braket_squared = braket**2
        omega_sq = ((omega + 1j * 0.02) * hbar)**2
        return 2 * e**2 * np.sum(braket_squared * E_nm / (E_nm**2 - omega_sq))

    omega_values = np.linspace(1, 5, 5000)

    # Sum contributions from both spin channels
    alpha_X = np.array([
        alpha_xx(omega, braketx_up,   eigvals_up[:num_up],     eigvals_up[num_up:])   +
        alpha_xx(omega, braketx_down, eigvals_down[:num_down],  eigvals_down[num_down:])
        for omega in omega_values
    ])
    alpha_Y = np.array([
        alpha_xx(omega, brakety_up,   eigvals_up[:num_up],     eigvals_up[num_up:])   +
        alpha_xx(omega, brakety_down, eigvals_down[:num_down],  eigvals_down[num_down:])
        for omega in omega_values
    ])

    alpha_values = (alpha_X + alpha_Y) / 2  # average over x and y (2D molecule)

    hbar_SI = 1.054571800e-34
    c = 299792458e9
    wavelength = c / ((hbar * omega_values * 1.60218e-19) / (2 * np.pi * hbar_SI))

    plt.plot(wavelength, np.imag(alpha_values), c='r', label='Im')
    plt.plot(wavelength, np.real(alpha_values), c='b', label='Re')
    plt.legend()
    plt.grid()
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'$\alpha$ [A$^2$ s$^4$ kg$^{-1}$]')
    plt.xlim(250, 800)
    plt.show()


s = 0 #1/2
plot_polarisability(1.930, s)
