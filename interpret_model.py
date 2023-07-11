# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:36:56 2022

@author: 18326
"""

from HeccLib import HecFeatureGenerator, HecPropertyParser, get_models
import joblib, os
import json

# interpret models
ANN_models = get_models()
pca_model   = joblib.load(os.path.join('checkpoint',
                                          'pca_model.joblib'))

with open('input_config.json', 'r') as f:
    config = json.load(f)

hfg = HecFeatureGenerator(prop_precursor_path=config['prop_precursor_path'],
                          props=config['props'],
                          operators=config['operators'])
hpp = HecPropertyParser(fname='HECC_properties_over_sample.CSV')
formula = hpp.get_property_from_str('real_formula')
x_data = hfg.get_input_feature_from_formula(*formula, scale_factor=1.3,
                                   save_log=False,
                                   load_log=True,
                                   log_file=os.path.join('checkpoint',
                                                         'scale_range.json'))
x_data = pca_model.transform(x_data)

con = hfg.get_concentration_from_ase(*formula)
mixing_entropy = hfg.get_mixing_entropy(*con) # / 14 # the maximum entropy of HEC with 5 metallics are 13.3816715.
ave, std, span = hfg.get_ave_std_span_prop(*con)
features = np.hstack([np.reshape(mixing_entropy, (-1, 1)), ave, std, span])
max_list = np.max(features, axis=0)
min_list = np.min(features, axis=0)
perturbate_magnitude = 0.2 * (max_list - min_list)

for i in range(37):
    features_tmp = features.copy()
    features_tmp[:, i] += perturbate_magnitude[i]
    features_tmp = hfg.scale_features(features_tmp, 1.3, False, True, os.path.join('checkpoint', 'scale_range.json'))
    features_tmp = np.where(features_tmp < 1e-14, 1e-14, features_tmp)
    features_expand = hfg.expand_features(features_tmp)
    features_expand = hfg.scale_features(features_expand, 1.3,
                                          False, True,
                                          os.path.join('checkpoint',
                                                       'scale_range_1.json'))

    features_tmp = np.hstack([features_tmp, features_expand])
    x_data = pca_model.transform(features_tmp)

    predictions_all = []
    for model in ANN_models:
        predictions = model.predict([x_data])
        predictions_all.append(predictions)

    predictions_all = np.array(predictions_all)
    prediction_mean = np.mean(predictions_all, axis=0)
    predictions_all = np.concatenate(predictions_all, axis=1)

    np.savetxt(os.path.join('interpret', f'y_data_{i}.txt'), prediction_mean, fmt='%.8f')

# compare
y_init = np.loadtxt(os.path.join('interpret', 'y_data_init.txt'))

variance = []
for i in range(37):
    y_tmp = np.loadtxt(os.path.join('interpret', f'y_data_{i}.txt'))
    diff = (y_tmp - y_init) / y_init * 100 # unit: %
    mean = np.mean(diff, axis=0)
    variance.append(mean)
variance = np.array(variance)
with open(os.path.join('interpret', 'variance.txt'), 'w') as f:
    print('B    G    E    Hv    C11    C44', file=f)
    np.savetxt(f, variance, fmt='%.8f')

# plot
import matplotlib.pyplot as plt
plt.bar(range(37), variance[:,6])
