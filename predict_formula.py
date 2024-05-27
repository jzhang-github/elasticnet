# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:51:55 2024

@author: 18326
"""

from elasticnet import predict_formula
import numpy as np

formulas = np.loadtxt('formulas.txt', dtype='str')

pf = predict_formula(config='input_config.json',ckpt_file='checkpoint')
r = pf.predict(*formulas)
np.savetxt('ML_results.txt', r, fmt='%.8f')
