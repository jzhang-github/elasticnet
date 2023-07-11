# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:45:11 2022

@author: 18326
"""

from HeccLib import predict_formula, get_concentration_from_ase, get_number_of_components
import sys
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    pf = predict_formula(config='input_config.json',
                         ckpt_file='checkpoint')

    elements = ['Ti', 'Nb', 'Ta']
    formulas, cons = [], []

    grid_num = 31
    # generate formulas
    for a in range(0, grid_num):
        end = grid_num - a
        for b in range(0, end):
            c = grid_num - a - b - 1
            formula = f'{elements[0]}{a}{elements[1]}{b}{elements[2]}{c}'
            _, formula = get_number_of_components(formula)
            if _ > 0:
                formulas.append(formula)
                cons.append([a/(grid_num - 1), b/(grid_num - 1), c/(grid_num - 1)])

    prediction_mean = pf.predict(*formulas)
    result = np.hstack([cons, prediction_mean])

    # save results
    df = pd.DataFrame(result, columns=elements+['B','G','E','Hv','C11','C44'])
    if not os.path.exists('phase_diagrams'):
        os.mkdir('phase_diagrams')
    df.to_csv(os.path.join('phase_diagrams', f'{"-".join(elements)}_diagram.csv'))