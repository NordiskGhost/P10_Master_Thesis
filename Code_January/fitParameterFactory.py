import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.constants import physical_constants, h, k
from collections import defaultdict
import scipy.optimize as sciop


################################ Structures
CC_BOND = 1.42
CC_CUTOFF = 1.6
CH_BOND = 1.09


def anthrancene():
    xyz = {}
    counter = np.arange(3)
    for i in counter:
        for j in range(6):
            theta = j * np.pi / 3 + (np.pi/6) # The 30deg is to turn the hexagons on its side
            x, y, z = np.cos(theta)+np.sqrt(3)/2+i*np.sqrt(3)/2, np.sin(theta)+i*3/2, 0
            xyz[f"{x}, {y}, {z}"] = [x, y, z]
    nri_list = ['Test']
    xyz_array = np.array(list(xyz.values()))
    while len(nri_list) > 0:
        nri_list = []
        for nri, i in enumerate(xyz_array):
            for nrj, j in enumerate(xyz_array):
                if nri == nrj:
                    continue
                else:
                    if np.isclose(i,j, atol=0.03).all() == np.array([True, True, True]).all():
                        nri_list.append(nri)
        if not nri_list:
            break
        del_val = nri_list[0]
        nri_list = [k for k in nri_list if k != del_val]
        xyz_array = np.delete(xyz_array, del_val, 0)
    return xyz_array * CC_BOND

def n_triangulene(N):
    xyz = {}
    counter = np.arange(N)
    for i in counter:
        for j in range(6):
            theta = j * np.pi / 3 + (np.pi/6) # The 30deg is to turn the hexagons on its side
            x, y, z = np.cos(theta)+np.sqrt(3)/2+i*np.sqrt(3)/2, np.sin(theta)+i*3/2, 0
            xyz[f"{x}, {y}, {z}"] = [x, y, z]
            xyz[f"{-x+np.sqrt(3)}, {y}, {z}"] = [-x+np.sqrt(3), y, z]
            #plt.scatter(dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, color = 'black')
                    
    if N > 2:
        length = N-2
        dists = np.array(list(xyz.values()))
        for dist in dists:
            dist[0] += -np.sqrt(3)/2
            dist[1] += np.sqrt(3)/2
            if np.linalg.norm(dist) <= length + length*np.sqrt(3)/2:
                xyz[f"{dist[0]+np.sqrt(3)/2}, {3+dist[1]-np.sqrt(3)/2}, {z}"] = [dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, z]
                #plt.scatter(dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, color = 'b')
    if N > 4:
        length = N-2.25
        dists = np.array(list(xyz.values()))
        for dist in dists:
            dist[0] += -np.sqrt(3)/2
            dist[1] += np.sqrt(3)/2
            if np.linalg.norm(dist) <= length + length*np.sqrt(3)/2:
                xyz[f"{dist[0]+np.sqrt(3)/2}, {3+dist[1]-np.sqrt(3)/2}, {z}"] = [dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, z]
                #plt.scatter(dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, color = 'r')
    if N > 6:
        length = N-2.25
        dists = np.array(list(xyz.values()))
        for dist in dists:
            dist[0] += -np.sqrt(3)/2
            dist[1] += np.sqrt(3)/2
            if np.linalg.norm(dist) <= length + length*np.sqrt(3)/2:
                xyz[f"{dist[0]+np.sqrt(3)/2}, {3+dist[1]-np.sqrt(3)/2}, {z}"] = [dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, z]
                #plt.scatter(dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, color = 'g')
    #plt.show()
    
    # Remove duplicate points due to approximation errors
    nri_list = ['Test']
    xyz_array = np.array(list(xyz.values()))
    while len(nri_list) > 0:
        nri_list = []
        for nri, i in enumerate(xyz_array):
            for nrj, j in enumerate(xyz_array):
                if nri == nrj:
                    continue
                else:
                    if np.isclose(i,j, atol=0.03).all() == np.array([True, True, True]).all():
                        nri_list.append(nri)
        if not nri_list:
            break
        del_val = nri_list[0]
        nri_list = [k for k in nri_list if k != del_val]
        xyz_array = np.delete(xyz_array, del_val, 0)

    return xyz_array * CC_BOND

def add_hydrogen(xyz):
    nrC = len(xyz)
    neighbors = [[] for _ in range(nrC)]
    for i in range(nrC):
        for j in range(i+1, nrC):
            dist = np.linalg.norm(xyz[i] - xyz[j])
            if dist < CC_CUTOFF:
                neighbors[i].append(j)
                neighbors[j].append(i)
    hydrogens = []
    for i, nbrs in enumerate(neighbors):
        coord = len(nbrs)
        missing = 3 - coord # sp2 carbon has three bonds in total
        if missing <= 0:
            continue
        ri = xyz[i]
        bond_vecs = np.zeros(3)
        for j in nbrs: # Sum of bond vectors to neighbors since the outward direction is opposite of bond sum
            bond_vecs += (xyz[j] - ri)
        if np.linalg.norm(bond_vecs) < 1e-6:
            continue
        direction = -bond_vecs / np.linalg.norm(bond_vecs)
        # Add hydrogen on the remaining missing bonds
        for _ in range(missing):
            h_pos = ri + CH_BOND * direction
            hydrogens.append(h_pos)
    return np.array(hydrogens)
################################

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

def Hubbard(U, S_z):
    # Hubbard parameter U [eV]
    global xyz
    delta1 = 2.7
    delta2 = 0.20
    delta3 = 0.18
    
    # Spinless Hamiltonian H0
    N = len(xyz) # Num. sites and electrons (at half-filling)  
    H0 = np.zeros((N, N))
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
    #print(n_up_new)
    #print(n_down_new)
    spin_den = n_up_new - n_down_new
    return H_total, total_energy, spin_den

def readCoordinates(path):
    #coords = {}
    if '7AGNR' in path:
        range_1 = 98
        range_2 = 35
        coords_arr = np.zeros((range_1,3))
        hydro_arr = np.zeros((range_2,3))
    elif 'indene' in path:
        range_1 = 7
        range_2 = 9
        coords_arr = np.zeros((range_2,3))
        hydro_arr = np.zeros((range_1,3))
    elif 'indeno' in path:
        range_1 = 12
        range_2 = 20
        coords_arr = np.zeros((range_2,3))
        hydro_arr = np.zeros((range_1,3))
    elif 'fluorene' in path:
        range_1 = 9
        range_2 = 13
        coords_arr = np.zeros((range_2,3))
        hydro_arr = np.zeros((range_1,3))
    with open(f'ORCA_files/{path}.txt', 'r') as f:
        for line in f:
            #if 'Number of atoms' in line:
            #    nr_atoms = int(line.split()[-1])
            if 'Coordinates' in line:
                break
        if '7AGNR' in path:
            for nr in range(range_1):
                next_line = f.readline().split()
                coords_arr[nr] = np.array([float(next_line[2]), float(next_line[3]), float(next_line[4])])
            for nr in range(range_2):
                next_line = f.readline().split()
                hydro_arr[nr] = np.array([float(next_line[2]), float(next_line[3]), float(next_line[4])])
        else:
            for nr in range(range_1):
                next_line = f.readline().split()
                hydro_arr[nr] = np.array([float(next_line[2]), float(next_line[3]), float(next_line[4])])
            for nr in range(range_2):
                next_line = f.readline().split()
                coords_arr[nr] = np.array([float(next_line[2]), float(next_line[3]), float(next_line[4])])
    return coords_arr, hydro_arr

def loadAlternative(name):
    with open(f'Reordered/{name}_hyp.txt', 'r') as f:
        CA_read = []
        HA_read = []
        is_HA = False
        for line in f:
            if not '[' in line:
                if 'Hydrogen' in line:
                    is_HA = True
                continue
            firstLine = line.replace('[','').replace(']','')
            secondLine = f.readline().replace('[','').replace(']','')
            thirdLine = f.readline().replace('[','').replace(']','')
            if is_HA == False:
                CA_read.append([[float(x) for x in firstLine.split()], [float(x) for x in secondLine.split()], [float(x) for x in thirdLine.split()]])
            else:
                HA_read.append([[float(x) for x in firstLine.split()], [float(x) for x in secondLine.split()], [float(x) for x in thirdLine.split()]])
        CA_read = np.array(CA_read)
        HA_read = np.array(HA_read)
    if '7AGNR' in name:
        return CA_read[1:], HA_read[-34:]
    else:
        return CA_read, HA_read
    
def loadHypFile(n):
    with open(f'Reordered/reordered_{n}triangulene_hyp.txt', 'r') as f:
        CA_read = []
        HA_read = []
        is_HA = False
        for line in f:
            if not '[' in line:
                if 'Hydrogen' in line:
                    is_HA = True
                continue
            firstLine = line.replace('[','').replace(']','')
            secondLine = f.readline().replace('[','').replace(']','')
            thirdLine = f.readline().replace('[','').replace(']','')
            if is_HA == False:
                CA_read.append([[float(x) for x in firstLine.split()], [float(x) for x in secondLine.split()], [float(x) for x in thirdLine.split()]])
            else:
                HA_read.append([[float(x) for x in firstLine.split()], [float(x) for x in secondLine.split()], [float(x) for x in thirdLine.split()]])
        CA_read = np.array(CA_read)
        HA_read = np.array(HA_read)
    return CA_read, HA_read

def hyp_A(spin_density, xyz, S, DC0, DC1, FC0, FC1):
    xyz = xyz
    dim = len(spin_density)
    Q0 = np.diag((-1, -1, 2))/2
    distances = np.linalg.norm(xyz[:, np.newaxis] - xyz, axis=2)
    A_dip = []
    A_iso = []
    for i in range(dim):
        #nearest_dist = np.sort(distances[i])[1]
        mask = (distances[i] < 1.7) & (distances[i] > 1e-6)
        mask[i] = False
        nearest_spin_density = spin_density*mask
        A_iso.append(FC0*spin_density[i]+FC1*np.sum(nearest_spin_density))
        distances2 = xyz[:, np.newaxis] - xyz
        sub_dist_arr = (distances2[i]) * mask[:, None]  #.T * mask).T
        Qvm = np.zeros((dim, 3, 3))
        for j, vm in enumerate(sub_dist_arr):
            prod = 3 * np.outer(vm, vm)
            len_vm = np.linalg.norm(vm)            
            if len_vm < 0.0001:
                continue
            iden = np.diag((1,1,1))
            Qvm[j] = (prod - len_vm**2*iden)/len_vm**5
        summa = np.zeros((3,3))
        for k in range(dim):
            summa += Qvm[k] * nearest_spin_density[k]
        A_dip.append(DC0*Q0*spin_density[i]+DC1*summa)
    A_iso = np.array(A_iso)/(2*S)
    A_dip = np.array(A_dip)/(2*S)
    return A_iso, A_dip

def hydro_clean(spin_density, xyz, hydro_xyz, S, FH1, DH0, DH1):
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

def sciCarbon(x):
    global spin_density, CA_hyp, CA
    FC0, FC1, DC0, DC1 = x[0], x[1], x[2], x[3]
    A_iso, A_total = hyp_A(spin_density, xyz, S, DC0, DC1, FC0, FC1)
    test_lis = []
    for j in (CA_hyp - A_total):
        test_lis.append(np.linalg.norm(j)**2)    
    return np.sum(np.array(test_lis)) + np.sum(abs(CA - A_iso)**2) 

def sciHydro(x):
    global spin_density, hydro_xyz, HA_hyp, HA, xyz
    FC1, DC0, DC1 = x[0], x[1], x[2]
    A_iso, A_total = hydro_clean(spin_density, xyz, hydro_xyz, S, FC1, DC0, DC1)
    test_lis = []
    for j in (HA_hyp - A_total):
        test_lis.append(np.linalg.norm(j)**2)
    return np.sum(np.array(test_lis)) + np.sum(abs(HA - A_iso)**2)

def sciHydroGlobal(x):
    FC1, DC0, DC1 = x[0], x[1], x[2]
    total_loss = 0.0
    for mol_data in all_hydro_data:
        spin_den, xyz_m, hydro_xyz_m, HA_hyp_m, HA_m, S_m = mol_data
        A_iso, A_total = hydro_clean(spin_den, xyz_m, hydro_xyz_m, S_m, FC1, DC0, DC1)
        mol_loss = 0.0
        for j in (HA_hyp_m - A_total):
            mol_loss += np.linalg.norm(j)**2
        mol_loss += np.sum(abs(HA_m - A_iso)**2)
        total_loss += mol_loss / len(HA_m)  # normalize per molecule
    return total_loss

if __name__ == "__main__":
    carbon_list = []
    hydro_list = []
    
    carbon_list_alt = []
    hydro_list_alt = []
    
    all_hydro_data = []
    
    U = 4.0 #3.0, 1.930
    if True: #Only 7AGNR, indene, indeno[2,1-b]fluorene*, fluorene
        for name in ['7AGNR', 'fluorene', 'indeno_21b_fluorene']: #, 'indene']:
            print(':'*11, name, ':'*11)
            xyz, hydro_xyz = readCoordinates(f'{name}_B3LYP_epr_property')
            S = 1/2
            if 'indeno' in name:
                S = 1
            CA_hyp, HA_hyp = loadAlternative(name)        
            
            #for nr, i in enumerate(xyz):
            #    plt.scatter(i[1], i[0], color = 'red')
            #    plt.annotate(f'{nr}', (i[1], i[0]))
            #for nr, i in enumerate(hydro_xyz):
            #    plt.scatter(i[1], i[0], color = 'blue')
            #    plt.annotate(f'{nr}', (i[1], i[0]))
            #plt.show()
            
            CA = np.zeros(len(CA_hyp))
            for nr, i in enumerate(CA_hyp):
                CA[nr] = np.trace(i)/3
                CA_hyp[nr] = i - (np.trace(i)/3) * np.eye(3)
            HA = np.zeros(len(HA_hyp))
            for nr, i in enumerate(HA_hyp):
                HA[nr] = np.trace(i)/3
                HA_hyp[nr] = i - (np.trace(i)/3) * np.eye(3)
            if '7AGNR' in name:
                xyz = np.delete(xyz, 1, axis=0)
                HA = np.delete(HA, 1)
                HA_hyp = np.delete(HA_hyp, 1, 0)
            _, _, spin_den = Hubbard(U, S)
            print('Spin densities:', spin_den)
            print('Total spin:', sum(spin_den))
            
            all_hydro_data.append((spin_den, xyz, hydro_xyz, HA_hyp, HA, S))
            
            if True:
                    spin_density = spin_den
                    if U > 3.5:
                        bon = [(80, 95), (-40, -20), (160, 200), (0, 10)]
                        x0 = [86, -33, 173, 5]
                    elif 3.5 > U > 2.5:
                        bon = [(98, 113), (-49, -35), (180, 210), (0, 30)]
                        x0 = [104, -42, 196, 21]
                    elif U < 1.7:
                        bon = [(133, 150), (-64, -48), (220, 240), (37, 53)]
                        x0 = [142, -53, 235, 44]
                    else:
                        bon = [(120, 150), (-75, -25), (180, 270), (15, 60)]
                        x0 = [132, -54, 223, 39]
                    result = sciop.minimize(sciCarbon, x0, method='L-BFGS-B', bounds= bon)
                    print(result)
                    print((result.x[0]-86.7)/86.7, (result.x[1]-(-33.1))/-33.1)
                    print((result.x[2]-173.3)/173.3, (result.x[3]-(4.6))/4.6)
                    carbon_list_alt.append(result.x)
                    
                    print('='*50)
                    if U > 3.5:
                        bon = [(-90, -65), (-55, 385), (-860, 800)]
                        x0 = [-75, 20, 35]
                    elif 3.5 > U > 2.5:
                        bon = [(-100, -75), (-55, 55), (-100, 120)]
                        x0 = [-85, 20, 40]
                    elif U < 1.7:
                        bon = [(-120, -60), (15, 35), (30, 50)]
                        x0 = [-100, 20, 40]
                    else:
                        bon = [(-170, -25), (-100, 100), (-100, 100)]
                        x0 = [-120, 33, 55]
                    result = sciop.minimize(sciHydro, x0, method='L-BFGS-B', bounds= bon)
                    print(result)
                    print((result.x[0]-(-78.7))/-78.7)
                    print((result.x[1]-22.6)/22.6, (result.x[2]-35.3)/35.3)
                    hydro_list_alt.append(result.x)
    carbon_list_alt = np.array(carbon_list_alt)
    hydro_list_alt = np.array(hydro_list_alt)
    print('FC0:', np.mean(carbon_list_alt[:,0]))
    print('FC1:', np.mean(carbon_list_alt[:,1]))
    print('DC0:', np.mean(carbon_list_alt[:,2]))
    print('DC1:', np.mean(carbon_list_alt[:,3]))
    print('FH1:', np.mean(hydro_list_alt[:,0]))
    print('DH0:', np.mean(hydro_list_alt[:,1]))
    print('DH1:', np.mean(hydro_list_alt[:,2]))
    
    if True:
        for n in range(2,8):
            print(':'*11, n, ':'*11)
            S = (1/2) * (n-1)

            CA_hyp, HA_hyp = loadHypFile(n)
            hydro_xyz = add_hydrogen(n_triangulene(n))
            #for nr, i in enumerate(hydro_xyz):
            #    plt.scatter(i[1], i[0], color = 'blue')
            #    plt.annotate(f'{nr}', (i[1], i[0]))
            #plt.show()

            CA = np.zeros(len(CA_hyp))
            for nr, i in enumerate(CA_hyp):
                CA[nr] = np.trace(i)/3
                CA_hyp[nr] = i - (np.trace(i)/3) * np.eye(3)

            HA = np.zeros(len(HA_hyp))
            for nr, i in enumerate(HA_hyp):
                HA[nr] = np.trace(i)/3
                HA_hyp[nr] = i - (np.trace(i)/3) * np.eye(3)
            
            #CA, HA, CA_hyp, HA_hyp = CA, HA, CA_hyp, HA_hyp

            xyz = n_triangulene(n)
            _, _, spin_den = Hubbard(U, S)
            print('Spin densities:', spin_den)
            print('Total spin:', sum(spin_den))
            hydro_xyz = add_hydrogen(xyz)
            
            for nr, i in enumerate(HA_hyp):
                HA[nr] = np.trace(i)/3
                HA_hyp[nr] = i - (np.trace(i)/3) * np.eye(3)  # subtract isotropic part
            
            all_hydro_data.append((spin_den, xyz, hydro_xyz, HA_hyp, HA, S))
            
            if True:
                spin_density = spin_den
                if U > 3.5:
                    bon = [(80, 95), (-40, -20), (160, 200), (0, 10)]
                    x0 = [86, -33, 173, 5]
                elif 3.5 > U > 2.5:
                    bon = [(98, 113), (-49, -35), (180, 210), (0, 30)]
                    x0 = [104, -42, 196, 21]
                elif U < 1.7:
                    bon = [(133, 150), (-64, -48), (220, 240), (37, 53)]
                    x0 = [142, -53, 235, 44]
                else:
                    bon = [(120, 150), (-55, -45), (210, 230), (27, 60)]
                    x0 = [132, -54, 223, 39]
                result = sciop.minimize(sciCarbon, x0, method='L-BFGS-B', bounds= bon)
                print(result)
                print((result.x[0]-86.7)/86.7, (result.x[1]-(-33.1))/-33.1)
                print((result.x[2]-173.3)/173.3, (result.x[3]-(4.6))/4.6)
                carbon_list.append(result.x)
                
                print('='*50)
                if U > 3.5:
                    bon = [(-90, -65), (-55, 485), (-75, 860)]
                    x0 = [-75, 20, 35]
                elif 3.5 > U > 2.5:
                    bon = [(-100, -75), (-55, 55), (-75, 75)]
                    x0 = [-85, 20, 40]
                elif U < 1.7:
                    bon = [(-120, -60), (15, 35), (30, 50)]
                    x0 = [-100, 20, 40]
                else:
                    bon = [(-150, -45), (-100, 100), (-100, 100)]
                    x0 = [-120, 33, 55]
                result = sciop.minimize(sciHydro, x0, method='L-BFGS-B', bounds= bon)
                print(result)
                print((result.x[0]-(-78.7))/-78.7)
                print((result.x[1]-22.6)/22.6, (result.x[2]-35.3)/35.3)
                hydro_list.append(result.x)
        carbon_list = np.array(carbon_list)
        hydro_list = np.array(hydro_list)
        print('='*20 + ' Results ' + '='*20)

        print('FC0:', np.mean(carbon_list[:,0]))
        print('FC1:', np.mean(carbon_list[:,1]))
        print('DC0:', np.mean(carbon_list[:,2]))
        print('DC1:', np.mean(carbon_list[:,3]))
        print('FH1:', np.mean(hydro_list[:,0]))
        print('DH0:', np.mean(hydro_list[:,1]))
        print('DH1:', np.mean(hydro_list[:,2]))

print('='*20 + ' Avg. ' + '='*20)
print('FC0:', (np.mean(carbon_list[:,0])+np.mean(carbon_list_alt[:,0]))/2)
print('FC1:', (np.mean(carbon_list[:,1])+np.mean(carbon_list_alt[:,1]))/2)
print('DC0:', (np.mean(carbon_list[:,2])+np.mean(carbon_list_alt[:,2]))/2)
print('DC1:', (np.mean(carbon_list[:,3])+np.mean(carbon_list_alt[:,3]))/2)
print('FH1:', (np.mean(hydro_list[:,0]) +np.mean(hydro_list_alt[:,0]))/2)
print('DH0:', (np.mean(hydro_list[:,1]) +np.mean(hydro_list_alt[:,1]))/2)
print('DH1:', (np.mean(hydro_list[:,2]) +np.mean(hydro_list_alt[:,2]))/2)

print('='*20 + ' GLOBAL ' + '='*20)

result = sciop.minimize(sciHydroGlobal, x0=[-78.63, 28.2, 26.8], method='L-BFGS-B', bounds= [(-90, -65), (10, 40), (-115, 155)])
print(result)
print((result.x[0]-(-78.7))/-78.7)
print((result.x[1]-22.6)/22.6, (result.x[2]-35.3)/35.3)

DH0_range = np.linspace(10, 40, 50)
DH1_range = np.linspace(-15, 55, 50)
loss_grid = np.zeros((50, 50))
for i, dh0 in enumerate(DH0_range):
    for j, dh1 in enumerate(DH1_range):
        loss_grid[i, j] = sciHydro([-78.63, dh0, dh1])
plt.contourf(DH1_range, DH0_range, loss_grid, levels=50)
plt.colorbar()
plt.xlabel('DH1')
plt.ylabel('DH0')
plt.scatter([35.3], [22.6], color='red', marker='*', s=200, label="Paper values")
plt.scatter([26.8], [28.2], color='white', marker='*', s=200, label="Sim. minimum")
plt.legend()
plt.show()


for i, name in enumerate(['7AGNR', 'fluorene', 'indeno_21b_fluorene', 'indene']):
    spin_den, xyz_m, hydro_xyz_m, HA_hyp_m, HA_m, S_m = all_hydro_data[i]
    spin_density = spin_den
    xyz = xyz_m
    hydro_xyz = hydro_xyz_m
    HA_hyp = HA_hyp_m
    HA = HA_m
    S = S_m
    result = sciop.minimize(sciHydro, [-78.63, 28.2, 26.8], method='L-BFGS-B', bounds=[(-90,-65),(10,40),(10,55)])
    print(f"{name}: FH1={result.x[0]:.2f}, DH0={result.x[1]:.2f}, DH1={result.x[2]:.2f}")
for mol_data in all_hydro_data[:3]:
    spin_den, xyz_m, hydro_xyz_m, HA_hyp_m, HA_m, S_m = mol_data
    distances = np.linalg.norm(xyz_m[:, None] - xyz_m, axis=2)
    for i in range(len(xyz_m)):
        mask = (distances[i] < 1.7) & (distances[i] > 1e-6)
        onsite = abs(spin_den[i])
        neighbor_sum = np.sum(abs(spin_den[mask]))
        if onsite > 0.01:
            print(f"onsite/neighbor ratio: {onsite/neighbor_sum:.3f}")





"""
U = 1.930
### Only non-bipartite      ### All structures included
FC0: 124.74850871054713     FC0: 126.36428743619439
FC1: -50.844351526358786    FC1: -52.91467338608345
DC0: 220.5739721793977      DC0: 217.84461684675566
DC1: 35.11983901213598      DC1: 45.11235860145083
FH1: -98.02062342376149     FH1: -99.49357106051428

DH0: 36.4198615211043       DH0: 4.162333842816114
DH1: 27.092125452810226     DH1: -4.525808883113157



==================== Results ====================
FC0: 126.36428743619439
FC1: -52.91467338608345
DC0: 217.84461684675566
DC1: 45.11235860145083
FH1: -99.49363864012197
DH0: 29.673308537724324
DH1: 100.0
==================== Avg. ====================
FC0: 125.55639807337076
FC1: -51.87951245622112
DC0: 219.20929451307669
DC1: 40.1160988067934
FH1: -98.98156168503857
DH0: 64.83665426886216
DH1: 68.15212997154633




U = 2.0
==================== Avg. ====================
FC0: 124.40348874734005
FC1: -51.272038592970674
DC0: 217.77901770000506
DC1: 38.897824531086144
FH1: -98.15178478382083
DH0: 20.14550749276809
DH1: 11.27499734903116
"""