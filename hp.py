# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 23:34:40 2022

@author: 18326
"""
import numpy as np
import time
import os
import tensorflow as tf
from HeccLib import HecFeatureGenerator, HecFeatureParser, get_number_of_components, predict_formula
import joblib
from prepare_input import x_main
import json
import pandas as pd

# generate formulas
num_comp = 'all'
num, formulas = [], []
start = time.time()
step = 0
for Ti_c in range(0, 11):
    end = 11 - Ti_c
    for V_c in range(0, end):
        end = 11 - Ti_c - V_c
        for Cr_c in range(0, end):
            end = 11 - Ti_c - V_c - Cr_c
            for Zr_c in range(0, end):
                end = 11 - Ti_c - V_c - Cr_c - Zr_c
                for Nb_c in range(0, end):
                    end = 11 - Ti_c - V_c - Cr_c - Zr_c - Nb_c
                    for Mo_c in range(0, end):
                        end = 11 - Ti_c - V_c - Cr_c - Zr_c - Nb_c - Mo_c
                        for Hf_c in range(0, end):
                            end = 11 - Ti_c - V_c - Cr_c - Zr_c - Nb_c - Mo_c - Hf_c
                            for Ta_c in range(0, end):
                                W_c = 11 - Ti_c - V_c - Cr_c - Zr_c - Nb_c - Mo_c - Hf_c - Ta_c
                                # for W_c in range(0, end):
                                step += 1
                                formula = f'Ti{Ti_c}V{V_c}Cr{Cr_c}Zr{Zr_c}Nb{Nb_c}Mo{Mo_c}Hf{Hf_c}Ta{Ta_c}W{W_c}'

                                n_comp, real_formula = get_number_of_components(formula)
                                # if n_comp == num_comp:
                                # if n_comp > 1:
                                formulas.append(real_formula)
                                num.append(n_comp)
dur = time.time() - start
formulas = [x for x in formulas if x != '']
num = [x for x in num if x != 0]
# predict
pf = predict_formula()

prediction_mean = pf.predict(*formulas)

# save results
# np.savetxt(f'ht_predict_{num_comp}_metals.txt', prediction_mean, fmt='%.8f')
df0 = pd.DataFrame({
                    'formula' : formulas
                    })
df1 = pd.DataFrame(prediction_mean, columns=['B', 'G', 'E', 'Hv', 'C11', 'C44'])
df = pd.concat([df0, df1], axis=1)

df.to_csv('ANN_predictions.csv')
df_B = df.sort_values(by=['B'], ascending=False)
df_G = df.sort_values(by=['G'], ascending=False)
df_E = df.sort_values(by=['E'], ascending=False)
df_Hv = df.sort_values(by=['Hv'], ascending=False)
df_C11 = df.sort_values(by=['C11'], ascending=False)
df_C44 = df.sort_values(by=['C44'], ascending=False)

with pd.ExcelWriter('ANN_predictions.xlsx') as writer:
    df.to_excel(writer, sheet_name='all_pred')
    df_B.to_excel(writer, sheet_name='B')
    df_G.to_excel(writer, sheet_name='G')
    df_E.to_excel(writer, sheet_name='E')
    df_Hv.to_excel(writer, sheet_name='Hv')
    df_C11.to_excel(writer, sheet_name='C11')
    df_C44.to_excel(writer, sheet_name='C44')
