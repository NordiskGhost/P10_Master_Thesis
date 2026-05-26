import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
 

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

def n_triangulene(N):
    xyz = {}
    counter = np.arange(N)
    for i in counter:
        for j in range(6):
            theta = j * np.pi / 3 + (np.pi/6) # The 30deg is to turn the hexagons on its side
            x, y, z = np.cos(theta)+np.sqrt(3)/2+i*np.sqrt(3)/2, np.sin(theta)+i*3/2, 0
            xyz[f"{x}, {y}, {z}"] = [x, y, z]
            xyz[f"{-x+np.sqrt(3)}, {y}, {z}"] = [-x+np.sqrt(3), y, z]
                    
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
                #plt.scatter(dist[0]+np.sqrt(3)/2, 3+dist[1]-np.sqrt(3)/2, color = 'g')
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

    return xyz_array

CC_BOND = 1.42
CC_CUTOFF = 1.6
CH_BOND = 1.09

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

################################################################################
# Lattice points and number of electrons with fixed distances.
################################################################################

def plot_hubbard(nr_points):
    energy_list_singlet = np.zeros(nr_points)
    energy_list_triplet = np.zeros(nr_points)
    hubbard_U = np.linspace(0, 6, nr_points)
    count = 0
    for i in range(0, nr_points, 1):
        _, total_energy_triplet = Hubbard(hubbard_U[i], 1)
        _, total_energy_singlet = Hubbard(hubbard_U[i], 0)
        energy_list_singlet[count] = total_energy_singlet
        energy_list_triplet[count] = total_energy_triplet
        count += 1
    dist = energy_list_singlet-energy_list_triplet
    np.savetxt(f"S-T_dist{nr_points}", dist)
    plt.plot(hubbard_U, energy_list_singlet, color = 'blue', label="Singlet")
    #plt.scatter(hubbard_U, energy_list_singlet, color = 'blue', marker = '.')
    plt.plot(hubbard_U, energy_list_triplet, color = 'red', label ="Triplet")
    #plt.scatter(hubbard_U, energy_list_triplet, color = 'red', marker = '.')
    plt.grid()
    plt.legend()
    plt.xlabel(r'Hubbard $U$ [eV]')
    plt.ylabel(r'$\hbar\omega$ [eV]')
    plt.xlim(0,6)
    #plt.clf()
    plt.show()

### With Hubbard, to third nearest neighbors
def Hubbard(U, S_z):
    global xyz
    N = len(xyz) # Num. sites and electrons (at half-filling)
    
    # Hubbard parameter U [eV]
    delta1 = 2.7
    delta2 = 0.20
    delta3 = 0.18
    
    ##################################################################
    ### FOR THE SUPPLEMENTARY INFORMATION OF NATURE ARTICLE: https://www.nature.com/articles/s41557-025-01776-1
    # Spinless Hamiltonian H0
    H0 = np.zeros((N, N))
    
    first = np.zeros((N,N), dtype=int)
    second = np.zeros((N,N), dtype=int)
    third = np.zeros((N,N), dtype=int)
    distances = np.linalg.norm(xyz[:, np.newaxis] - xyz, axis=2)
    for i in range(N):
        nearest = mult_sort(distances[i])[1:4]
        nearest = [i[0] for i in nearest]
        first[i] = np.isclose(distances[i], nearest[0], atol=0.1)
        second[i] = np.isclose(distances[i], nearest[1], atol=0.1)
        #third[i] = np.isclose(distances[i], nearest[2], atol=0.1)
        #first[i] = np.isclose(distances[i], 1.42, atol=0.1) #1.00 1.42
        #second[i] = np.isclose(distances[i], 2.46, atol=0.1) #1.73 2.46
        third[i] = np.isclose(distances[i], 2.0, atol=0.1) #2.646 2.84
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
    n_up = np.ones(N) * num_up/N
    n_down = np.ones(N) * num_down/N
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
    return H_total, total_energy, n_up_new, n_down_new

def carbon_old(spin_density, xyz, S):
    leng = len(spin_density)
    FC0 = 86.7
    FC1 = -33.1
    DC0 = 173.3
    DC1 = 4.6 
    Q0 = np.diag((-1, -1, 2))/2
    distances = np.linalg.norm(xyz[:, np.newaxis] - xyz, axis=2)
    A_iso = []
    A_dip = []
    A_total = np.zeros((leng,3,3))
    for i in range(leng):
        nearest_dist = np.sort(distances[i])[1]
        mask = np.isclose(nearest_dist, distances[i], atol=0.1)
        nearest_spin_density = spin_density*mask
        A_iso.append(FC0*spin_density[i]+FC1*np.sum(nearest_spin_density))
        distances2 = xyz[:, np.newaxis] - xyz
        sub_dist_arr = (distances2[i].T * mask).T
        Qvm = np.zeros((leng, 3, 3))
        for j, vm in enumerate(sub_dist_arr):
            prod = 3 * vm.T @ vm
            len_vm = np.linalg.norm(vm)
            if -0.0001 < len_vm < 0.0001:
                continue
            iden = np.diag((1,1,1))
            Qvm[j] = (prod - len_vm**2*iden)/len_vm**5
        summa = np.zeros((3,3))
        for k in range(leng):
            summa += Qvm[k] * nearest_spin_density[k]
        A_dip.append(DC0*Q0*spin_density[i]+DC1*summa)
        print('here!', np.trace(DC0*Q0*spin_density[i]+DC1*summa))
    A_iso = np.array(A_iso)#/(2*S)
    A_dip = np.array(A_dip)#/(2*S)
    for l in range(leng):
        A_total[l] = A_iso[l] * np.diag((1,1,1)) + A_dip[l]
    return A_iso, A_total


def carbon(spin_density, xyz, S):
    dim = len(spin_density)
    Q0 = np.diag((-1, -1, 2))/2
    FC0 = 86.7
    FC1 = -33.1
    DC0 = 173.3
    DC1 = 4.6 
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
            if len_vm < 0.0001:
                continue
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

def hydro(spin_density, xyz, hydro_xyz, S): #C-H sigma orbitals!! Or maybe an additional interaktion of a C-13
    leng = len(hydro_xyz)
    FH0 = 0
    FH1 = -78.7
    DH0 = 22.6
    DH1 = 35.3
    Q0 = np.diag((-1, -1, 2))/2
    A_iso = []
    A_dip = []
    A_total = np.zeros((leng,3,3))
    
    for i in range(leng):
        for j in range(len(xyz)):
            dist = hydro_xyz[i] - xyz[j]
            if np.linalg.norm(dist) <= CH_BOND + 0.02:
                key = j
                break
        A_iso.append(FH0*0.00+FH1*spin_density[key])

        dist = dist.reshape((3,1))
        prod = 3 * dist @ dist.T
        len_vm = np.linalg.norm(dist)
        iden = np.diag((1,1,1))
        Qvm = (prod - len_vm**2*iden)/(len_vm**5)
        A_dip.append(DH0*Q0*spin_density[key] + DH1*Qvm*spin_density[key])
    A_iso = np.array(A_iso)#/(2*S)
    A_dip = np.array(A_dip)#/(2*S)
    for l in range(leng):
        A_total[l] = A_iso[l] * np.diag((1,1,1)) + A_dip[l]
    return A_iso, A_total

def t2_times(A_total, spin_densities):
    P = np.mean(spin_densities)
    length = len(A_total)
    summa = 0
    for i in range(length):
        summa += A_total[i][2][2]**2
    sigma = np.sqrt(((1-P**2)/4)*abs(summa))
    T2time = np.sqrt(2)/sigma
    return sigma, T2time*1e3

if __name__ == "__main__":
    n = 'anthrancene'
    xyz = anthrancene()
    #xyz = n_triangulene(n)
    #xyz = generateGoblet()
    #xyz = np.array([[1.420000000000, 0.000000000000, 0.000000000000],[2.130000000000, 1.229756070000, 0.000000000000],[2.130000000000, -1.229756070000, 0.000000000000],[3.550000000000, 1.229756070000, 0.000000000000],[3.550000000000, -1.229756070000, 0.000000000000],[4.260000000000, 2.459512150000, 0.000000000000],[4.260000000000, 0.000000000000, 0.000000000000],[4.260000000000, -2.459512150000, 0.000000000000],[5.680000000000, 2.459512150000, 0.000000000000],[5.680000000000, 0.000000000000, 0.000000000000],[6.390000000000, 1.229756070000, 0.000000000000],[6.390000000000, -1.229756070000, 0.000000000000],[5.680000000000,-2.459512150000, 0.000000000000]])
    #xyz = np.array([[1.446098488693, 0.000000030666, -0.000000619905],[2.130443794457, 1.206691905396, 0.000000422237],[2.130443722678, -1.206692112693, 0.000000400147],[3.546936934448, 1.237983640278, 0.000000065168],[3.546936942311, -1.237983657983, -0.000000189309],[4.269918079058, 2.442468947103, 0.000000083152],[4.260161071015, -0.000000036016, -0.000000550255],[4.269918078387, -2.442468850684, 0.000000169214],[5.674545457543, 2.466552338831, 0.000000453508],[5.680100079534, -0.000000123149, 0.000000262798],[5.674545492870, -2.466552211604, 0.000000604377],[6.409769130481, 3.677721031320, -0.000001037525],[6.389918767393, 1.229763697525, 0.000000102777],[6.389918853018, -1.229763799856, 0.000000101930],[6.409769088457, -3.677720845547, -0.000000855185],[7.796966842049, 3.667139845758, 0.000000273522],[7.818730729149, 1.228561069434, 0.000000939672],[7.818730908846, -1.228560987941, 0.000000599718],[7.796966729538, -3.667139820288, 0.000000254320],[8.499704980206, 2.471017146859, -0.000000237322],[8.500071797413, -0.000000008322, 0.000000389209],[8.499704920426, -2.471017072555, -0.000000383945]])
    
    if n == 'anthrancene':
        S = 1/2
    else:
        S = (n-1)/2
    print("N:", len(xyz))
    plt.grid()
    plt.scatter(xyz[:, 0], xyz[:, 1])
    plt.axis('equal')
    plt.show()
    _, tot_energy, n_up, n_down = Hubbard(4.00, S)
    print('Spin densities:', n_up-n_down)
    
    N = len(n_up)
    x = np.linspace(1, N, N)
    plt.plot(x,n_up, color = 'b')
    plt.plot(x,n_down, color = 'r')
    plt.plot(x,(n_up-n_down), color = 'g')
    plt.clf()

    print('='*50)
    print('Carbon-13')
    print('='*50)
    CA_iso, CA_total = carbon(n_up-n_down, xyz, S)
    print('A_iso:')
    print(CA_iso) #*2*S
    print('A_total tensor:')
    print(CA_total)
    if False:
        with open(f'{n}triangulene_hyp.txt', 'w') as f:
            f.write('='*20 + 'Carbon' + '='*20 + '\n')
            f.write(np.array2string(CA_total))
    #print(CA_total)
    #eigvals, eigvecs = np.linalg.eigh(CA_total)
    #print(eigvals)
    #binwidth = 5.2
    #hist_list = []
    #for i in CA_total:
    #    hist_list.append(np.round(i[2][2], 9))
    #plt.hist(hist_list, bins=np.arange(min(hist_list), max(hist_list) + binwidth, binwidth))
    #plt.show()
    
    ##################################################
    print('='*50)
    print('Hydrogen-1')
    print('='*50)
    hydro_xyz = add_hydrogen(xyz)
    carbon_xyz = xyz
    xyz = hydro_xyz
    plt.grid()
    plt.scatter(xyz[:, 0], xyz[:, 1])
    plt.axis('equal')
    plt.show()
    HA_iso, HA_total = hydro(n_up-n_down, carbon_xyz, hydro_xyz, S)
    if False:
        with open(f'{n}triangulene_hyp.txt', 'a') as f:
            f.write('\n' + '='*19 + 'Hydrogen' + '='*19 + '\n')
            f.write(np.array2string(HA_total))
    print('A_iso:')
    print(HA_iso) #*2*S
    print('A_total tensor eigenvalues:')
    eigvals, eigvecs = np.linalg.eigh(HA_total)
    print(eigvals)
    #for i in eigvals:
        #print(max(abs(i))) #*2*S
    #hist_list = []
    #for i in HA_total:
    #    hist_list.append(np.round(i[2][2], 9))
    #print(hist_list)
    #plt.hist(hist_list, bins=np.arange(min(hist_list), max(hist_list) + binwidth, binwidth))
    #plt.show()
    print('Hydrogen - T2* times') # for C12 structure
    sigma, T2time = t2_times(HA_total, n_up-n_down)
    print(sigma)
    print(T2time)
    print('='*50)
    print('Total T2* times')
    print('='*50)
    A_total = np.append(CA_total, HA_total, axis=0)
    sigma, T2time = t2_times(A_total, n_up-n_down)
    print(sigma)
    print(T2time)


    if False:
        with open(f'{n}triangulene_hyp.txt', 'r') as f:
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
        print('CA', CA_read)
        print('HA', HA_read)
