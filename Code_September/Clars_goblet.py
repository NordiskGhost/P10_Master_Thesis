import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
 
def generateGoblet():
    ### Turn and duplicate triangulene structure to form Clar's goblet
    lower = generateTriangulene()
    upper = lower.copy()
    upper[:, 1] = -upper[:, 1]+1
    whole = np.concatenate((lower, upper), axis = 0)
    return whole

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

def dos(bandsList, bins):
    energies = np.concatenate(bandsList)
    print(len(bandsList[0]))
    plt.hist(energies, bins=bins, density=True)
    plt.xticks(np.arange(-12, 13, 3))
    plt.xlabel('Energy eigenvalues [eV]')
    plt.ylabel('Density of states (arb. units)')
    plt.grid(True)
    plt.show()

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
            pass
    return sorted_items

################################################################################
# Lattice points and number of electrons with fixed distances.
################################################################################
#xyz = generateGoblet()
#N = len(xyz) #Num. electrons
################################################################################
# Lattice points form supplementary
with open("Clar_supplement.txt", encoding='utf-8') as file:
    supp_data = []
    while True:
        content = file.readline()
        if "C " in content:
            supp_data.append(content[2:])
        if not content:
            break

C_data = np.zeros((len(supp_data), 3))

count = 0
for line in supp_data:
    temp = [float(i) for i in line.split(' ')]
    C_data[count] = temp
    count += 1

xyz_supp = C_data[:76]
singlet_xyz = xyz_supp[:38]
triplet_xyz = xyz_supp[38:76]

if False:
    plt.grid()
    plt.scatter(xyz[:, 1], xyz[:, 0])
    plt.scatter(singlet_xyz[:, 0], singlet_xyz[:, 1])
    plt.scatter(triplet_xyz[:, 0], triplet_xyz[:, 1])
    plt.axis('equal')
    plt.show()

N = len(singlet_xyz) #Num. electrons
################################################################################

### With Hubbard, to third nearest neighbors
def Hubbard(U, S_z):
    global xyz
    if S_z == 0:
        xyz = singlet_xyz
    elif S_z == 1:
        xyz = triplet_xyz
    
    # Hubbard parameter U [eV]
    delta1 = 2.7
    delta2 = 0.20
    delta3 = 0.18
    
    ##################################################################
    ### FOR THE SUPPLEMENTARY INFORMATION OF NATURE ARTICLE: https://www.nature.com/articles/s41557-025-01776-1
    # Spinless Hamiltonian H0
    H0 = np.zeros((N, N))
    #print(N)
    
    first = np.zeros((N,N), dtype=int)
    second = np.zeros((N,N), dtype=int)
    third = np.zeros((N,N), dtype=int)
    distances = np.linalg.norm(xyz[:, np.newaxis] - xyz, axis=2)
    for i in range(38):
        nearest = mult_sort(distances[i])[1:4]
        nearest = [i[0] for i in nearest]
        first[i] = np.isclose(distances[i], nearest[0], atol=0.1)
        second[i] = np.isclose(distances[i], nearest[1], atol=0.1)
        third[i] = np.isclose(distances[i], nearest[2], atol=0.1)
    
    H0 += -delta1*first-delta2*second-delta3*third
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

    #n_up = np.full(N, 0.5)-1e7 # Inital guess
    #n_down = np.full(N, 0.5)+1e7
    n_up = np.ones(N) * num_up/N -1e-7
    n_down = np.ones(N) * num_down/N +1e-7
    alpha = 0.2

    # The self-consistent loop
    for iteration in range(1000):
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
        tolerance = 1e-14
        if np.max(np.abs(n_up - n_up_new)) < tolerance and np.max(np.abs(n_down - n_down_new)) < tolerance:
            print(f"Converged after {iteration} iterations. S = {S_z}")
            #print(H_total) # Sanity check
            break
        elif iteration == 1500:
            print(f"Max iteration, did not converge. S = {S_z}")
        # Linear comb. for faster convergence. Uses the electron densities from both new and old n.
        n_up = alpha * n_up_new + (1 - alpha) * n_up
        n_down = alpha * n_down_new + (1 - alpha) * n_down
    
    E_kinetic = np.sum(eigvals_up[:num_up]) + np.sum(eigvals_down[:num_down])
    E_interaction = -U * np.sum((n_up_new-1/2) * (n_down_new-1/2))
    total_energy = E_kinetic + E_interaction/2
    #print("tot:", total_energy)
    #print("kin:", E_kinetic)
    #print("U:", E_interaction)
    return H_total, total_energy

#Hubbard(2.0, 0)

def plot_hubbard(nr_points):
    energy_list_singlet = np.zeros(nr_points)
    energy_list_triplet = np.zeros(nr_points)
    hubbard_U = np.linspace(0, 4, nr_points)
    count = 0
    for i in range(0, nr_points, 1):
        _, total_energy_triplet = Hubbard(hubbard_U[i], 1)
        _, total_energy_singlet = Hubbard(hubbard_U[i], 0)
        energy_list_singlet[count] = total_energy_singlet
        energy_list_triplet[count] = total_energy_triplet
        count += 1
    dist = energy_list_singlet-energy_list_triplet
    #np.savetxt(f"S-T_dist{nr_points}", dist)
    plt.plot(hubbard_U, energy_list_singlet, color = 'blue', label="Singlet")
    #plt.scatter(hubbard_U, energy_list_singlet, color = 'blue', marker = '.')
    plt.plot(hubbard_U, energy_list_triplet, color = 'red', label ="Triplet")
    #plt.scatter(hubbard_U, energy_list_triplet, color = 'red', marker = '.')
    plt.grid()
    plt.legend()
    plt.xlabel(r'Hubbard $U$ [eV]')
    plt.ylabel(r'$E_{tot}$ [eV]')
    plt.xlim(0,4)
    #plt.clf()
    plt.show()

plot_hubbard(750)



def plot_polarisability(U, S_z):
    H_total,_ = Hubbard(U, S_z)
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
    
    braketx = c_n.conjugate().T @ X_op @ c_m # |<n|x|m>|^2
    brakety = c_n.conjugate().T @ Y_op @ c_m # |<n|y|m>|^2
    braketz = c_n.conjugate().T @ Z_op @ c_m # |<n|z|m>|^2
    
    hbar = 1
    e = 1
    def alpha_xx(omega, braket):
        E_nm = eigenvalues[occupied:] - eigenvalues[:occupied, np.newaxis]
        braket_squared = braket**2
        omega_squared = (omega + 1j * 0.02) * hbar
        return 2 * e**2 * np.sum(braket_squared * E_nm / (E_nm**2 - omega_squared**2))
    
    omega_values = np.linspace(1, 5, 5000) # 1/s
    
    alpha_Xvalues = np.array([alpha_xx(omega, braketx) for omega in omega_values])
    alpha_Yvalues = np.array([alpha_xx(omega, brakety) for omega in omega_values])
    alpha_Zvalues = np.array([alpha_xx(omega, braketz) for omega in omega_values])

    alpha_values = 1/3 * (alpha_Xvalues + alpha_Yvalues + alpha_Zvalues)
    
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
    plt.plot(wavelength, np.imag(alpha_values), c='r')
    plt.plot(wavelength, np.real(alpha_values), c='b')
    plt.legend(["Im", "Re"])
    plt.grid()
    plt.xlabel(r'$\lambda$ [nm]')
    plt.ylabel(r'$\alpha$ [A$^2$ s$^4$ kg$^{-1}$]')
    plt.xlim(250,800)
    
    #plt.clf()
    plt.show()
    
    ### DOS (discrete energies for the 38 lattice sites)
    bands = []
    eig_list = list(zip(eigenvalues,eigenvectors))
    bands.append(np.array([i[0] for i in eig_list]).real)
    #dos(bands, 250)

#plot_polarisability(2.0, 0)

if False:
    plt.grid()
    plt.scatter(singlet_xyz[:, 0], singlet_xyz[:, 1])
    plt.axis('equal')
    plt.show()