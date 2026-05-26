import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.constants import physical_constants, h, k
from collections import defaultdict
import pandas as pd

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
#xyz = generateGoblet()
twist = False
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
    
    theta = 37 # degrees
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
    
    # FOR CATION: N - 1, FOR NAPTHALENE: N + 1, FOR NEUTRAL: N
    num_electrons = N +1 # Assume one electron per site (half-filling)
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

#FOR CATION AND NAPHTHALENE s = 1/2, AND FOR NEUTRAL s = 0
s = 1/2 # 0 , 1/2
_, tot_energy, spin_den = Hubbard(2.0, s)
print(tot_energy)
print('Spin densities:', spin_den)
print('Total spin:', sum(spin_den))


f = plt.figure()
f.set_figwidth(4.8)
f.set_figheight(5.2)
if len(xyz) < 12:
    f.set_figwidth(4)
    f.set_figheight(2)
elif (twist == True) and (s == 0):
    plt.title(r'$\theta = 37^{\circ}$, Neutral')
elif (twist == True) and (s == 1/2):
    plt.title(r'$\theta = 37^{\circ}$, Cation')
elif (twist == False) and (s == 1/2):
    plt.title(r'$\theta = 0^{\circ}$, Cation')
else:
    plt.title(r'$\theta = 0^{\circ}$, Neutral')

cmap = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
sc_plot = plt.scatter(xyz[:, 0], xyz[:, 1], marker = '.', s=300,  color = cmap(norm(spin_den)), edgecolors = 'black')
plt.axis('equal')
cbar = plt.colorbar()
cbar.set_label('Spin density', rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.axis('off')
sc_plot.set_cmap('jet')
plt.clim(-1.0,1.0)
for i, pos in enumerate(xyz):
    plt.annotate(f'{round((spin_den)[i], 3)}', (pos[0]-0.3, pos[1]+0.35), fontsize = 7.5)
plt.xlim(-1.6, 6.4) #-1.6, 6.4, Naph: -1.6, 4.0
plt.ylim(-5.4, 7.05) #-5.4, 7.05, Naph: -1.7, 2.05
plt.show()


###### Hyperfine values
# U = 1.930
#==================== Avg. ====================
# FC0: 125.55639807337076
# FC1: -51.87951245622112
# DC0: 219.20929451307669
# DC1: 40.1160988067934
# FH1: -98.75709724213789
# DH0: 29.673308537724324
# DH1: 68.15212997154633

# U = 4.0
# FC0: 86.63047445450992
# FC1: -33.366678229941684
# DC0: 173.50752316719561
# DC1: 4.315726046736708
# FH1: -78.19369030329531
# DH0: 20.97659516793622
# DH1: ??? 35.3

# U = 3.0
# FC0: 105.46994982653861
# FC1: -43.156767063003706
# DC0: 195.22384429810532
# DC1: 23.245775376398907
# FH1: -89.27938995837535
# DH0: 24.7390912334283
# DH1: 75.0
# ==================== Avg. ====================
# FC0: 104.64222003703767
# FC1: -42.60443538853931
# DC0: 196.60393925265444
# DC1: 21.483150241941793
# FH1: -89.05149340530251
# DH0: 39.86954561671415
# DH1: -12.5


def carbon(spin_density, xyz, S):
    #           (3.0),  (4.0), (1.930)
    FC0 = 86.63047 #104.9  #86.7  #125.556
    FC1 = -33.3667 #-42.3 #-33.1 #-51.8795
    DC0 = 173.508 #196.5  #173.3 #219.2093
    DC1 = 4.31573 #21.9   #4.6   #40.116098
    Q0 = np.diag((-1, -1, 2))/2
    dim = len(spin_density)
    distances = np.linalg.norm(xyz[:, np.newaxis] - xyz, axis=2)
    A_dip = []
    A_iso = []
    A_total = np.zeros((dim,3,3))
    for i in range(dim):
        #nearest_dist = np.sort(distances[i])[1]
        mask = (distances[i] < 1.7) & (distances[i] > 1e-6) #np.isclose(nearest_dist, distances[i], atol=0.5)
        mask[i] = False
        nearest_spin_density = spin_density*mask
        A_iso.append(FC0*spin_density[i]+FC1*np.sum(nearest_spin_density))
        distances2 = xyz[:, np.newaxis] - xyz
        sub_dist_arr = (distances2[i]) * mask[:, None]  #.T * mask).T
        Qvm = np.zeros((dim, 3, 3))
        for j, vm in enumerate(sub_dist_arr):
            prod = 3 * np.outer(vm, vm)
            len_vm = np.linalg.norm(vm)
            if -0.0001 < len_vm < 0.0001:
                len_vm = 0.0001
            iden = np.diag((1,1,1))
            Qvm[j] = (prod - len_vm**2*iden)/len_vm**5
        summa = np.zeros((3,3))
        for k in range(dim):
            summa += Qvm[k] * nearest_spin_density[k]
        A_dip.append(DC0*Q0*spin_density[i]+DC1*summa)
    A_iso = np.array(A_iso)#/(2*S)
    A_dip = np.array(A_dip)#/(2*S)
    for l in range(dim):
        A_total[l] = A_iso[l] * np.diag((1,1,1)) + A_dip[l]
    return A_iso, A_total


# FH1: -98.75709724213789
# DH0: 20.29109768196021
# DH1: 11.283158284848534

# DH0: 29.673308537724324
# DH1: 68.15212997154633

# FH1: -78.19369030329531
# DH0: 20.97659516793622
# DH1: ??? 35.3

def hydro_test(spin_density, xyz, hydro_xyz, S):
    FH0 = 0                 #(1.930)  (3.0)       (4.0)
    FH1 = -78.67            #-98.7571 #-89.7     #-78.7
    DH0 = 25.36             #20.29110 #25.4      #22.6
    DH1 = 34.58              #11.28316 #40.5      #35.3
    Nh = len(hydro_xyz)
    distances_CC = np.linalg.norm(xyz[:, None] - xyz, axis=2)
    A_iso = []
    A_dip = []

    for i in range(Nh):
        r_H = hydro_xyz[i]
        dists_CH = np.linalg.norm(xyz - r_H, axis=1)
        key = np.argmin(dists_CH)
        if dists_CH[key] > CH_BOND + 0.2:
            continue
        A_iso.append(FH1 * spin_density[key])

        r_CH = (r_H - xyz[key])
        r_norm = np.linalg.norm(r_CH)
        if r_norm < 1e-10:
            Q_CH = np.zeros((3, 3))
        else:
            Q_CH = CH_BOND**3 * (3 * np.outer(r_CH, r_CH) - r_norm**2 * np.eye(3)) / r_norm**5
        A0 = DH0 * spin_density[key] * Q_CH
        
        neighbor_mask = (distances_CC[key] < 1.7) & (distances_CC[key] > 1e-6)
        neighbor_mask[key] = False
        
        A1 = np.zeros((3, 3))
        for m in np.where(neighbor_mask)[0]:
            r_Hm = (r_H - xyz[m])
            r_norm = np.linalg.norm(r_Hm)
            if r_norm < 1e-10:
                continue
            Q_Hm = CC_BOND**3 *(3 * np.outer(r_Hm, r_Hm) - r_norm**2 * np.eye(3)) / r_norm**5
            A1 += spin_density[m] * Q_Hm
        A1 *= DH1
        
        A_dip.append(A0 + A1)
    A_iso = np.array(A_iso) / (2 * S)
    A_dip = np.array(A_dip) / (2 * S)
    return A_iso, A_dip



print('='*50)
print('Carbon-13')
print('='*50)
CA_iso, CA_total = carbon(spin_den, xyz, 0)
print('A_iso:')
print(CA_iso) #*2*S

#with open(f'Goblet/Neutral_Goblet_hyp.txt', 'w') as f:
#    f.write('='*20 + 'Carbon' + '='*20 + '\n')
#    f.write(np.array2string(CA_total))


f = plt.figure()
f.set_figwidth(4.8)
f.set_figheight(5.2)
if len(xyz) < 12:
    f.set_figwidth(4)
    f.set_figheight(2)
elif (twist == True) and (s == 0):
    plt.title(r'$\theta = 37^{\circ}$, Neutral')
elif (twist == True) and (s == 1/2):
    plt.title(r'$\theta = 37^{\circ}$, Cation')
elif (twist == False) and (s == 1/2):
    plt.title(r'$\theta = 0^{\circ}$, Cation')
else:
    plt.title(r'$\theta = 0^{\circ}$, Neutral')

cmap = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=-40.0, vmax=40.0)
sc_plot = plt.scatter(xyz[:, 0], xyz[:, 1], marker = '.', s=300,  color = cmap(norm(CA_iso)), edgecolors = 'black')
plt.axis('equal')
cbar = plt.colorbar()
cbar.set_label(r'$A_0$ [MHz]', rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.axis('off')
sc_plot.set_cmap('jet')
plt.clim(-40.0, 40.0)
for i, pos in enumerate(xyz):
    plt.annotate(f'{round((CA_iso)[i], 3)}', (pos[0]-0.3, pos[1]+0.35), fontsize = 7.5)
plt.xlim(-1.6, 6.4) #-1.6, 6.4, Naph: -1.6, 4.0
plt.ylim(-5.4, 7.05) #-5.4, 7.05, Naph: -1.7, 2.05
plt.show()

print('A_total tensor:')
eigvals, eigvecs = np.linalg.eigh(CA_total)
print(eigvals)


print('='*50)
print('Hydrogen-1')
print('='*50)
hydro_xyz = add_hydrogen(xyz)
carbon_xyz = xyz
xyz = hydro_xyz
plt.grid()
plt.scatter(xyz[:, 0], xyz[:, 1])
plt.axis('equal')
plt.clf()
HA_iso, HA_total = hydro_test(spin_den, carbon_xyz, hydro_xyz, s)
print('A_iso:')
print(HA_iso) #*2*S
print(HA_total)

#with open(f'Goblet/Neutral_Goblet_hyp.txt', 'a') as f:
#    f.write('\n' + '='*19 + 'Hydrogen' + '='*19 + '\n')
#    f.write(np.array2string(HA_total))

f = plt.figure()
f.set_figwidth(4.8)
f.set_figheight(5.2)
if len(xyz) < 12:
    f.set_figwidth(4)
    f.set_figheight(2)
elif (twist == True) and (s == 0):
    plt.title(r'$\theta = 37^{\circ}$, Neutral')
elif (twist == True) and (s == 1/2):
    plt.title(r'$\theta = 37^{\circ}$, Cation')
elif (twist == False) and (s == 1/2):
    plt.title(r'$\theta = 0^{\circ}$, Cation')
elif len(xyz) < 12:
    pass
else:
    plt.title(r'$\theta = 0^{\circ}$, Neutral')

plt.scatter(carbon_xyz[:, 0], carbon_xyz[:, 1], marker = '.', s=75,  color = 'black')
cmap = plt.cm.jet
norm = matplotlib.colors.Normalize(vmin=-40.0, vmax=40.0)
sc_plot = plt.scatter(xyz[:, 0], xyz[:, 1], marker = '.', s=300,  color = cmap(norm(HA_iso)), edgecolors = 'black')
plt.axis('equal')
cbar = plt.colorbar()
cbar.set_label(r'$A_0$ [MHz]', rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.axis('off')
sc_plot.set_cmap('jet')
plt.clim(-40.0, 40.0)
for i, pos in enumerate(xyz):
    plt.annotate(f'{round((HA_iso)[i], 3)}', (pos[0]-0.75, pos[1]+0.35), fontsize = 7.5)
plt.xlim(-1.6, 6.4) #-1.6, 6.4, Naph: -3, 5
plt.ylim(-6.5, 8.1) #-5.4, 7.05, Naph: -3, 3
plt.show()

print('A_total tensor eigenvalues:')
eigvals, eigvecs = np.linalg.eigh(HA_total)
print(eigvals)








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
gammaL = 1.85*0.175004e-3    # Linewidth (Lorentzian, Tesla)
#gammaG = 0.24e-3            # Linewidth (technically the standard deviation, Tesla)

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
freq = 9.592e9
gammaL = 1.45*0.175004e-3  
deriv_spectrum = CWEPR(2, (12.46520342)*1e6, 2, (12.70779595)*1e6)
#deriv_spectrum2 = CWEPR(2, 12.1955e6, 2, 12.6311e6)

print((12.54593099-12.1955)/(12.1955), (12.7900946-12.6311)/(12.6311))

df = pd.read_excel('Fig4-plots.xlsx', sheet_name='4c', usecols='C:D')
df2 = pd.read_excel('Fig4-plots.xlsx', sheet_name='4c', usecols='N:O')
data = df.to_numpy()
data2 = df2.to_numpy()
#data[:, 1] = normalizeY(data[:, 1])


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
deriv_spectrum = (1e3/9)*CWEPR(2, 12.98998306e6, 2, 12.99629785e6) + 0.5#CWEPR(2, 17.001e6, 2, 16.8e6))
deriv_spectrum2 = (1e3/6.6)*CWEPR(2, 11.1245e6, 2, 14.5349e6) + 0.5


print('Lorentz gamma [MHz]:', round(gammaL * (g_e * mu_B / h)*1e-6, 3))
print("--- %s seconds ---" % (time.time() - start_time))
plt.plot(B0*1e3, deriv_spectrum, color='blue', label='Simulated')
#plt.plot(B0*1e3, deriv_spectrum2, color='red', label='Simulated')
plt.plot(data[:, 0], data[:, 1], color='black', label='Jiao et al.')
plt.xlabel("Magnetic Field [mT]")
plt.legend()
plt.ylabel("Intensity [a.u.]")
plt.grid()
plt.xlim(min(B0*1e3), max(B0*1e3))
plt.show()



### Naphthalene
print(':'*15 + ' Naphthalene ' + ':'*15)
print((18.84740672 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (4.660251 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.865e-4 * (g_e * mu_B / h)*1e-6))
print((15.36366209 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (4.95547196 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.865e-4 * (g_e * mu_B / h)*1e-6))
print((15.07088417 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (5.10883427 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.865e-4 * (g_e * mu_B / h)*1e-6))
print((15.218 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (4.978 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.865e-4 * (g_e * mu_B / h)*1e-6))
print((14.78 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (5.261 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.865e-4 * (g_e * mu_B / h)*1e-6))


print((19.22 - (4.95e-4 * (g_e * mu_B / h)*1e-6))/(4.95e-4 * (g_e * mu_B / h)*1e-6), (6.287 - 1.865e-4 * (g_e * mu_B / h)*1e-6)/(1.865e-4 * (g_e * mu_B / h)*1e-6))

print((4.95e-4 * (g_e * mu_B / h)*1e-6), 1.865e-4 * (g_e * mu_B / h)*1e-6)

def plot_exp_naphthalene(B_offset):
    df = pd.read_csv('plot-data.csv')
    data = df.to_numpy()
    x, y = data[:, 0], data[:, 1]
    y /= y.max()
    plt.plot(B_offset + (x[:-1] - x[-1]), y[:-1], color = 'b', label = 'Weil & Bolten')
    #plt.show()

gammaL = 0.05e-4        # Linewidth (Lorentzian, Tesla)
B0 = np.linspace(0.336, 0.349, 10000)
freq = 9.59e9 
#deriv_spectrum = CWEPR(4, 14.78e6, 4, 5.261e6) # U = 2.0
deriv_spectrum = CWEPR(4, 15.218e6, 4, 4.978e6)
#deriv_spectrum2 = CWEPR(4, 0.495e-3 * (g_e * mu_B / h), 4, 0.1865e-3 * (g_e * mu_B / h)) # Weil & Boltons values, but not the actual plot
#deriv_spectrum2 = CWEPR(4, 14.923e6, 4, 5.186e6) # U = 1.4
print(0.495e-3 * (g_e * mu_B / h), 0.187e-3 * (g_e * mu_B / h))

deriv_spectrum /= deriv_spectrum.max()
#deriv_spectrum2 /= deriv_spectrum2.max()

print('Lorentz gamma [MHz]:', round(gammaL * (g_e * mu_B / h)*1e-6, 3))
print("--- %s seconds ---" % (time.time() - start_time))

plot_exp_naphthalene(1e3*B0[np.argmin(deriv_spectrum)])
plt.plot(B0*1e3, deriv_spectrum, color='r', label = r'Simulated $U = 4.0$')
plt.xlabel("Magnetic Field [mT]")
plt.ylabel("Intensity [a.u.]")
plt.legend()
plt.grid()
plt.xlim(340.56, 343.8)
plt.show()
