# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:20:54 2022

@author: 18326
"""

import numpy as np
import pandas as pd
from ase.formula import Formula
from HeccLib import get_number_of_components

contributions = {
    'Ti': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'V': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Cr': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Zr': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Nb': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Mo': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Hf': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Ta': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'W': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0}
                 }

concentrations = {
    'Ti': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'V': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Cr': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Zr': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Nb': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Mo': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Hf': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'Ta': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0},
    'W': {'B': 0, 'G': 0, 'E': 0, 'Hv': 0}
                 }

def get_concentration_from_ase(*formula: str):
    con = []
    for f in formula:
        f_dict = Formula(f).count()
        tot = np.sum(list(f_dict.values()))
        c_dict = {k: v/tot for k, v in f_dict.items() if v != 0.0 and k!='C'}
        con.append(c_dict)
    return con

df = pd.read_csv('ML_formula_moduli.CSV')
cons = get_concentration_from_ase(*df['formula'])
for row, c in enumerate(cons):
    for e in c:
        for prop in ['B', 'G', 'E', 'Hv']:
            contributions[e][prop] += c[e] * df[prop][row]
            concentrations[e][prop] += c[e]

num = len(df)
contributions1 = {ks:{k:v/concentrations[ks][k] for k,v in vs.items()} for ks, vs in contributions.items()}

df_results = pd.DataFrame(contributions1)
df_results.to_csv('elemental_contributions_ML.csv')
