# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:02:12 2022

@author: 18326
"""

import numpy as np
from HeccLib import HecFeatureGenerator, HecPropertyParser
import json
from ase.formula import Formula

'''
Dependent files
---------------
input_config.json: 输入参数文件
HECC_properties.CSV： 获取二元和多主元碳化物的信息
formulas_{metal_num}metals.txt： 包含化学式的文件
'''

metal_num = 2

with open('input_config.json', 'r') as f:
    config = json.load(f)

config['HECC_properties_path'] = 'HECC_properties MP binary for ROM only.CSV'

# get properties of HECC
hpp = HecPropertyParser(config['HECC_properties_path'])

formulas = np.loadtxt(f'formulas_{metal_num}metals.txt', dtype=str)

B_HECC = hpp.get_property_from_str('B')
G_HECC = hpp.get_property_from_str('G')
E_HECC = hpp.get_property_from_str('E')
real_formula_HECC = hpp.get_property_from_str('real_formula')

props = {k:{'B':B_HECC[i], 'G':G_HECC[i], 'E':E_HECC[i]} for i, k in enumerate(real_formula_HECC)}

B_HECC = [props[x]['B'] for x in formulas]
G_HECC = [props[x]['G'] for x in formulas]
E_HECC = [props[x]['E'] for x in formulas]

# get properties of precursors
hfg = HecFeatureGenerator(prop_precursor_path=config['prop_precursor_path'],
                              props = config['props'],
                              operators=config['operators'])

B_ROM, G_ROM, E_ROM = [], [], []
for f in formulas:
    # eles = list(Formula(f).count().keys())
    cons = hfg.get_concentration_from_ase(f)
    rom = [cons[0][x] * float(props[x]['B']) for x in cons[0]]
    B_ROM.append(np.sum(rom))
    rom = [cons[0][x] * float(props[x]['G']) for x in cons[0]]
    G_ROM.append(np.sum(rom))
    rom = [cons[0][x] * float(props[x]['E']) for x in cons[0]]
    E_ROM.append(np.sum(rom))

B_result = np.vstack([B_ROM, B_HECC]).T
G_result = np.vstack([G_ROM, G_HECC]).T
E_result = np.vstack([E_ROM, E_HECC]).T

np.savetxt(f'B_result_MP_new_{metal_num}.txt', B_result, fmt='%s')
np.savetxt(f'G_result_MP_new_{metal_num}.txt', G_result, fmt='%s')
np.savetxt(f'E_result_MP_new_{metal_num}.txt', E_result, fmt='%s')

B_diff = np.array(B_HECC, dtype=float) - np.array(B_ROM)
G_diff = np.array(G_HECC, dtype=float) - np.array(G_ROM)
E_diff = np.array(E_HECC, dtype=float) - np.array(E_ROM)
