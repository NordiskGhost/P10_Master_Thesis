import numpy as np
import matplotlib.pyplot as plt
import os

structure_files = os.listdir('ORCA_files')

def readFile(path):
    file_dict = {}
    with open(f'ORCA_files/{path}') as fp:
        for _ in range(88):
            next(fp)
        for line in fp:
            if 'Nucleus' in line:
                name = str(line.split()[-2:][1] + line.split()[-2:][0])
                for _ in range(5):
                    fp.readline()
                tensor = np.array([[float(x) for x in fp.readline().split()][-3:], [float(x) for x in fp.readline().split()][-3:], [float(x) for x in fp.readline().split()][-3:]])
                file_dict[f'{name}'] = tensor
            if 'Coordinates' in line:
                break
        for key in file_dict.keys():
            coords = [float(x) for x in fp.readline().split()[-3:]]
            file_dict[key] = (file_dict[key], coords)
    print(f'Read file: {path}')
    return file_dict

def hypDicts(file_dict):
    C_hyp = []
    H_hyp = []
    for key in file_dict.keys():
        if key[0] == 'C':
            C_hyp.append(file_dict[key][0])
        else:
            H_hyp.append(file_dict[key][0])
    return np.array(C_hyp), np.array(H_hyp)


def reorder_hyperfine_tensors(tensors, reference_iso, S, atol_start=1e-1, atol_max=10.0, atol_factor=0.1, warnings = True):
    def get_iso(tensor, S):
        return (2*S) * np.trace(np.array(tensor))/3
    available = [(i, get_iso(t, S), t) for i, t in enumerate(tensors)]
    reordered = []
    #print(len(available), len(reference_iso))
    #print(sum([bool(x[1]>0) for x in available]), sum([bool(x>0) for x in reference_iso]))
    for ref_val in reference_iso:
        atol = atol_start
        best_idx = None
        best_diff = np.inf

        while best_idx is None:
            counter = 0
            for j, (orig_i, iso, tensor) in enumerate(available):
                diff = abs(iso - ref_val)
                if diff <= atol and diff < best_diff:
                    best_diff = diff
                    best_idx = j
                counter += 1
            if best_idx is None:
                if atol >= atol_max:
                    raise ValueError(
                        f"No unused tensor found with isotropic value ≈ {ref_val:.4f} even at max tolerance {atol_max}.\n"
                        f"Remaining available iso values: {[round(x[1], 4) for x in available]}.\n"
                        f"Remaining ref. values: {[round(x, 4) for x in reference_iso[-counter:]]}."
                    )
                atol += atol_factor
        if atol > atol_start:
            matched_iso = available[best_idx][1]
            if warnings == True:
                print(f"Warning: Matched ref. value {ref_val:.4f} with {matched_iso:.4f} on elevated tolerance of atol={atol:.4e}")
        reordered.append(available.pop(best_idx)[2])
    return np.array(reordered)



if False:
    with open('reference_files/reference_lists', 'w') as f:
        f.write('='*20 + ' Isotropic Hyperfine Reference List ' + '='*20 + '\n'*2)
    for n in range(2,8):
        n = 'anthrancene'
        with open(f'reference_files/{n}triangulene_hyp.txt', 'r') as f:
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

            CA_ref_lis = []
            HA_ref_lis = []
            for nr, i in enumerate(CA_read):
                CA_ref_lis.append(np.trace(i)/3)
            for nr, i in enumerate(HA_read):
                HA_ref_lis.append(np.trace(i)/3)
        with open('reference_files/reference_lists', 'a') as f:
            f.write('='*20 + f' {n}triangulene CA ' + '='*20 + '\n')
            f.write(' '.join([str(x) for x in CA_ref_lis]) + '\n'*2)
            f.write('='*20 + f' {n}triangulene HA ' + '='*20 + '\n')
            f.write(' '.join([str(x) for x in HA_ref_lis]) + '\n'*2)


#print(hypDicts(readFile(structure_files[0]))[1])
### Loads ORCA B3LYP raw hyperfine tensors calculated by Sengupta et al.

ref_lists = {}
with open('reference_files/reference_lists', 'r') as f:
    count = 0
    for line in f:
        if not 'triangulene' in line:
            continue
        name = line[21] + line[34:36]
        ref_lists[f'{name}'] = [float(x) for x in f.readline().split()]


for n in range(2,8):
    print('\n' + f'{n}triangulene:::' + '\n')
    S = (n-1)/2
    with open(f'Reordered/reordered_{n}triangulene_hyp.txt', 'w') as f:
        f.write('='*20 + 'Carbon' + '='*20 + '\n')
        f.write(np.array2string(reorder_hyperfine_tensors(hypDicts(readFile(structure_files[n-2]))[0], ref_lists[f'{n}CA'], S, warnings= False)))
    with open(f'Reordered/reordered_{n}triangulene_hyp.txt', 'a') as f:
        f.write('\n' + '='*19 + 'Hydrogen' + '='*19 + '\n')
        f.write(np.array2string(reorder_hyperfine_tensors(hypDicts(readFile(structure_files[n-2]))[1], ref_lists[f'{n}HA'], S, warnings= False)))

