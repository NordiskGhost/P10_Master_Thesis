import numpy as np


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

"""
print(len(readCoordinates('7AGNR_B3LYP_epr_property')[0]),len(readCoordinates('7AGNR_B3LYP_epr_property')[1]))
print('\n' + '7AGNR:::' + '\n')
S = 1/2
with open(f'Reordered/7AGNR_hyp.txt', 'w') as f:
    f.write('='*20 + 'Carbon' + '='*20 + '\n')
    f.write(np.array2string(hypDicts(readFile('7AGNR_B3LYP_epr_property.txt'))[0]))
with open(f'Reordered/7AGNR_hyp.txt', 'a') as f:
    f.write('\n' + '='*19 + 'Hydrogen' + '='*19 + '\n')
    f.write(np.array2string(hypDicts(readFile('7AGNR_B3LYP_epr_property.txt'))[1]))
"""

####################################################################

for name in ['indene', 'fluorene', 'indeno_21b_fluorene']:
    print(':'*20, name, ':'*20)
    print(len(readCoordinates(f'{name}_B3LYP_epr_property')[0]),len(readCoordinates(f'{name}_B3LYP_epr_property')[1]))
    S = 1/2
    with open(f'Reordered/{name}_hyp.txt', 'w') as f:
        f.write('='*20 + 'Carbon' + '='*20 + '\n')
        f.write(np.array2string(hypDicts(readFile(f'{name}_B3LYP_epr_property.txt'))[0]))
    with open(f'Reordered/{name}_hyp.txt', 'a') as f:
        f.write('\n' + '='*19 + 'Hydrogen' + '='*19 + '\n')
        f.write(np.array2string(hypDicts(readFile(f'{name}_B3LYP_epr_property.txt'))[1]))


