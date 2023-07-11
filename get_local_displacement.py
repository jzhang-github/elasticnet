# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 15:31:31 2023

@author: ZHANG Jun
"""

from ase.io import read
import os
import pandas as pd
import numpy as np
from ase.formula import Formula
import matplotlib.pyplot as plt

def shift_fcoords2(fcoord1, fcoord2, cutoff=0.5):
    """ Relocate fractional coordinate to the image of reference point. ``fcoord1`` is the reference point.

    :param fcoord1: the reference point
    :type fcoord1: numpy.ndarry
    :param fcoord2: coordinate will be shifted
    :type fcoord2: numpy.ndarry
    :param cutoff: cutoff difference to wrap two coordinates to the same periodic image.
    :type cutoff: float
    :return: new ``fcoord2`` in the same periodic image of the reference point.
    :rtype: numpy.ndarry

    .. Important:: Coordinates should be ``numpy.ndarray``.

    """
    shift_status = False
    diff        = fcoord1 - fcoord2
    transition  = np.where(diff >= cutoff, 1.0, 0.0)
    if np.isin(1.0, diff):
        shift_status = True
    fcoord2_new = fcoord2 + transition
    transition  = np.where(diff < -cutoff, 1.0, 0.0)
    if np.isin(1.0, diff):
        shift_status = True
    fcoord2_new = fcoord2_new - transition
    return fcoord2_new, shift_status

def get_atomic_diameter(ase_atoms, crystal_type='fcc'):
    """ Calculate atomic diameter of a bulk structure.

    :param ase_atoms: input structure.
    :type ase_atoms: ase.atoms.Atoms
    :param crystal_type: crystal type, defaults to 'fcc'. Other options: 'bcc', 'hcp', and 'cubic'.
    :type crystal_type: str, optional
    :return: atomic diameter
    :rtype: float

    """
    atomic_density = {'fcc'  : 0.7404804896930611,
                        'bcc'  : 0.6801747615878315,
                        'hcp'  : 0.7404804896930611,
                        'cubic': 0.5235987755982988}

    cell_volume = ase_atoms.get_volume()
    num_sites   = len(ase_atoms)
    diameter    = (cell_volume * atomic_density[crystal_type] / num_sites * 3 / 4 / np.pi) ** (1 / 3) * 2
    return diameter

def get_centroid(fcoords, ref_pos, cutoff=0.5, convergence=0.00001):
    """

    :param fcoords: DESCRIPTION
    :type fcoords: TYPE
    :param ref_pos: DESCRIPTION
    :type ref_pos: TYPE
    :param cutoff: DESCRIPTION, defaults to 0.5
    :type cutoff: TYPE, optional
    :param convergence: DESCRIPTION, defaults to 0.00001
    :type convergence: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """
    fcoords_tmp = fcoords.copy()
    num_coord = np.shape(fcoords_tmp)[0]

    for i in range(num_coord):
        fcoords_tmp[i], shift_status = shift_fcoords2(ref_pos, fcoords_tmp[i], cutoff=cutoff)

    centroid_tmp = np.sum(fcoords_tmp, axis=0) / num_coord
    return centroid_tmp

def fractional2cartesian(vector_tmp, D_coord_tmp):
    """

    :param vector_tmp: DESCRIPTION
    :type vector_tmp: TYPE
    :param D_coord_tmp: DESCRIPTION
    :type D_coord_tmp: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    C_coord_tmp = np.dot(D_coord_tmp, vector_tmp)
    return C_coord_tmp

def get_local_displacement(ase_atoms):
    """

    :param ase_atoms: DESCRIPTION
    :type ase_atoms: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    scaled_positions = ase_atoms.get_scaled_positions()
    cell_array = ase_atoms.get_cell().array
    atomic_diameter = get_atomic_diameter(ase_atoms, crystal_type='cubic')
    cutff_for_centroid = atomic_diameter / np.mean(np.diagonal(cell_array)) * 1.25

    dist_matrix = ase_atoms.get_all_distances(mic=True)

    seqs = np.argsort(dist_matrix, axis=0)
    NN_indices = seqs[1:7,:]

    local_distortion = []
    for i in range(len(ase_atoms)):
        ref_pos = scaled_positions[i]

        NNi = NN_indices[:, i]
        NN_scaled_positions = scaled_positions[NNi]
        scaled_centroid = get_centroid(NN_scaled_positions, ref_pos=ref_pos, cutoff=cutff_for_centroid)


        scaled_centroid, status = shift_fcoords2(ref_pos, scaled_centroid, cutoff=0.5)

        cart_atoms = fractional2cartesian(cell_array, np.vstack((scaled_centroid, ref_pos)))

        dist = np.linalg.norm(cart_atoms[0,:]-cart_atoms[1,:], ord=2)

        local_distortion.append(dist)
    return local_distortion

def get_real_chemical_symbols(formula, ase_atoms):
    """

    :param formula: DESCRIPTION
    :type formula: TYPE
    :param ase_atoms: DESCRIPTION
    :type ase_atoms: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    """
    elements = list(Formula(formula).count())
    nums = list(Formula(ase_atoms.get_chemical_formula()).count().values())
    nums.pop(0)
    nums.append(32)
    symbols = [[x]*nums[i] for i, x in enumerate(elements)]
    symbols = [item for sublist in symbols for item in sublist]
    return symbols

def exclude_elements(df_data: pd.core.frame.DataFrame, exclude_eles_set):
    """

    :param df_data: DESCRIPTION
    :type df_data: pd.core.frame.DataFrame
    :param exclude_eles_set: DESCRIPTION
    :type exclude_eles_set: TYPE
    :return: DESCRIPTION
    :rtype: TYPE

    .. Note: This keys of results_data should contain 'real_formula'.

    """
    mask_code = [] # when True, this sample does not have elements listed in ele_list.
    for f in df_data['real_formula']:
        eles = set(Formula(f).count().keys())
        mask_code.append(not bool(eles&exclude_eles_set))
    new_df = df_data[mask_code]
    return new_df

def get_concentration_from_ase(*formula: str):
    # for carbides only
    # the C element will be depressed in this function.
    con = []
    for f in formula:
        f_dict = Formula(f).count()
        if f_dict.__contains__('C'):
            del f_dict['C']
        tot = np.sum(list(f_dict.values()))
        c_dict = {k+'C': v/tot for k, v in f_dict.items() if v != 0.0}
        con.append(c_dict)
    return con

def get_ROM(real_formula, props):
    # props is a dict
    # The unit given in the csv file of volume is volume_per_atom, but in reality, the unit should be volume_per_formula
    cons = get_concentration_from_ase(real_formula)
    B_rom = np.sum([cons[0][x] * float(props[x]['B']) for x in cons[0]])
    G_rom = np.sum([cons[0][x] * float(props[x]['G']) for x in cons[0]])
    E_rom = np.sum([cons[0][x] * float(props[x]['E']) for x in cons[0]])
    Hv_rom = np.sum([cons[0][x] * float(props[x]['Hv']) for x in cons[0]])
    V_rom = np.sum([cons[0][x] * float(props[x]['volume_per_atom']) for x in cons[0]])
    return B_rom, G_rom, E_rom, Hv_rom, V_rom

def get_deviation_from_ROM(real_formula, props):
    B_rom, G_rom, E_rom, Hv_rom, V_rom = get_ROM(real_formula, props)
    B, G, E, Hv, V = props[real_formula]['B'], props[real_formula]['G'], props[real_formula]['E'], props[real_formula]['Hv'], props[real_formula]['volume_per_atom']
    return B - B_rom, G - G_rom, E - E_rom, Hv - Hv_rom, V - V_rom

def get_volume_misfit(real_formula, props):
    # The unit given in the csv file of volume is volume_per_atom, but in reality, the unit should be volume_per_formula
    # volume misfits according equation (1) in https://doi.org/10.1016/j.mattod.2023.02.012
    cons = get_concentration_from_ase(real_formula)
    Valloy = props[real_formula]['volume_per_atom']
    v_diff = []
    for con in cons[0]:
        Vn = props[con]['volume_per_atom']
        delta_V = Vn-Valloy
        v_diff.append(cons[0][con] * delta_V ** 2)
    sigma = np.sum(v_diff) / (3 * Valloy)
    return sigma

def pad_dict_list(dict_list, padel):
    # https://stackoverflow.com/questions/40442014/python-pandas-valueerror-arrays-must-be-all-same-length
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

# read formulas
formulas = np.loadtxt('formulas_2-5metals.txt', dtype=str)

# read HECC properties
HECC_data = pd.read_csv('ANN_elastic_constants - recalculate.CSV')
HECC_data_dict = HECC_data.to_dict(orient='index')
HECC_data_dict = {HECC_data_dict[k]['real_formula']:HECC_data_dict[k] for k in HECC_data_dict}

# =============================================================================
# read CONTCARs and store local distortions to each element.
# =============================================================================
ele_distortion = dict(zip(['Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W', 'C'],
                          [[],[],[],[],[],[],[],[],[],[]]))
for formula in formulas:
    print(formula)
    eles = list(Formula(formula).count())
    fname = os.path.join('CONTCARs_new_calc', f'CONTCAR_{"".join(eles).rstrip("C")}')
    ase_atoms = read(fname)
    symbols = get_real_chemical_symbols(formula, ase_atoms)

    local_distortion = get_local_displacement(ase_atoms)

    for i, s in enumerate(symbols):
        ele_distortion[s].append(local_distortion[i])

ele_distortion = pad_dict_list(ele_distortion, None)
df_ele_distortion = pd.DataFrame(data=ele_distortion)
df_ele_distortion.to_csv('elements_distortion.csv')

# # plot
# for ele in ele_distortion:
#     plt.hist(ele_distortion[ele], bins=30)
#     plt.xlabel('Local distortion (angstrom)')
#     plt.ylabel('Frequency')
#     plt.xlim(0, 0.5)
#     plt.title(ele)
#     plt.savefig(f'{ele}_distortion.png')
#     plt.show()

# =============================================================================
# get local distortion of each cell and compare with elastic moduli
# =============================================================================
total_distortion_list = []
for formula in formulas:
    print(formula)
    eles = list(Formula(formula).count())
    fname = os.path.join('CONTCARs_new_calc', f'CONTCAR_{"".join(eles).rstrip("C")}')
    ase_atoms = read(fname)
    local_distortion = get_local_displacement(ase_atoms)
    total_distortion = np.sum(local_distortion)
    total_distortion_list.append(total_distortion)

# =============================================================================
# get deviations from ROM, including B, G, E, Hv, and volume
# =============================================================================
B_DFT_ROM, G_DFT_ROM, E_DFT_ROM, Hv_DFT_ROM, V_DFT_ROM = [], [], [], [], []
for formula in formulas:
    B_diff, G_diff, E_diff, Hv_diff, V_diff = get_deviation_from_ROM(formula, HECC_data_dict)
    B_DFT_ROM.append(B_diff)
    G_DFT_ROM.append(G_diff)
    E_DFT_ROM.append(E_diff)
    Hv_DFT_ROM.append(Hv_diff)
    V_DFT_ROM.append(V_diff)

# =============================================================================
# get volume misfits according equation (1) in https://doi.org/10.1016/j.mattod.2023.02.012
# =============================================================================
volume_misfits = [get_volume_misfit(x,HECC_data_dict)  for x in formulas]

# =============================================================================
# collect results
# =============================================================================
result = {'real_formula': formulas,
          'B': [HECC_data_dict[x]['B'] for x in formulas],
          'G': [HECC_data_dict[x]['G'] for x in formulas],
          'E': [HECC_data_dict[x]['E'] for x in formulas],
          'Hv': [HECC_data_dict[x]['Hv'] for x in formulas],
          'total_distortion': total_distortion_list,
          'B_DFT-ROM': B_DFT_ROM,
          'G_DFT-ROM': G_DFT_ROM,
          'E_DFT-ROM': E_DFT_ROM,
          'Hv_DFT-ROM': Hv_DFT_ROM,
          'Volume_expansion_V_DFT-ROM': V_DFT_ROM,
          'volume_misfits': volume_misfits
          }

df = pd.DataFrame(data=result)
df1 = exclude_elements(df, {'Cr', 'Mo', 'W'})
df2 = exclude_elements(df, {'Cr', 'Mo', 'W', 'V', 'Nb'})

# for prop in ['B', 'G', 'E']:
#     plt.scatter(df['total_distortion'], df[prop], label=f'{prop}_all')
#     plt.scatter(df1['total_distortion'], df1[prop], label=f'{prop}_no_CrMoW')
#     plt.scatter(df2['total_distortion'], df2[prop], label=f'{prop}_no_CrMoWVNb')
#     plt.legend()
#     plt.savefig(f'{prop}.png')
#     plt.show()

df.to_csv('all.csv')
df1.to_csv('no_CrMoW.csv')
df2.to_csv('no_CrMoWVNb.csv')
