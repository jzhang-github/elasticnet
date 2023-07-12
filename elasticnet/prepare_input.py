# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 21:57:03 2022

@author: ZHANG Jun
"""

import numpy as np
from elasticnet.lib import HecFeatureGenerator, HecPropertyParser, HecFeatureParser, FileTypeError, soap_feature
import json
import os

def x_main(config:dict, load_PCA=False, save_PCA=True):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    elif isinstance(config, dict):
        pass
    else:
        raise FileTypeError('The input config should be a name of json file or a `dict`.')

    if not os.path.exists(config["model_save_path"]):
        os.makedirs(config["model_save_path"])

    config = {
        'include_more': config['include_more']\
            if config.__contains__('include_more') else False,
        'split_test': config['split_test']\
            if config.__contains__('split_test') else False,
        'clean_by_pearson_r': config['clean_by_pearson_r']\
            if config.__contains__('clean_by_pearson_r') else False,
        'reduce_dimension_by_pca': config['reduce_dimension_by_pca']\
            if config.__contains__('reduce_dimension_by_pca') else True,
        'prop_precursor_path':  config['prop_precursor_path']\
            if config.__contains__('prop_precursor_path') else 'HECC_precursors.csv',
        'model_save_path':  config['model_save_path']\
            if config.__contains__('model_save_path') else 'checkpoint',
        'props': config['props']\
            if config.__contains__('props') else [# 'mixing_entropy', # the `HeccLib.HecFeatureGenerator.get_input_feature_from_formula` includes mixing_entropy automaticlly.
                                                  'volume_per_formula',
                                                  'm',
                                                  'rho',
                                                  'VEC',
                                                  'Pauling_X',
                                                  'Allen_X',
                                                  'group',
                                                  'periodic',
                                                  'C11',
                                                  'C12',
                                                  'C44',
                                                  'melting_point_of_sable_compound'],
        'operators': config['operators']\
            if config.__contains__('operators') else ['sqrt',
                                                      'log10',
                                                      'log',
                                                      'square',
                                                      'cube'],
        'HECC_properties_path': config['HECC_properties_path']\
            if config.__contains__('HECC_properties_path') else 'HECC_properties.CSV',
              }

    hfg = HecFeatureGenerator(prop_precursor_path=config['prop_precursor_path'],
                              props = config['props'],
                              operators=config['operators'])
    hpp = HecPropertyParser(config['HECC_properties_path'])

    # get x_data.txt
    formulas = hpp.get_property_from_str('real_formula').tolist()

    features = hfg.get_input_feature_from_formula(*formulas)
    con      = hfg.get_concentration_from_ase(*formulas)

    if config['include_more']:
        energies = hpp.get_property_from_str('bulk_energy_per_formula',
                                             dtype=float).tolist()
        volumes  = hpp.get_property_from_str('volume_per_formula',
                                             dtype=float).tolist()
        enthalpy = hfg.get_mixing_enthalpy(energies, con)
        v_diff   = hfg.get_volume_change(volumes, con)
        features = np.hstack((features,
                              np.array(enthalpy).reshape(-1,1),
                              np.array(v_diff).reshape(-1,1)))

    if config['split_test']:
        feat_tot = len(features)
        test_index = np.random.choice(range(feat_tot), size=int(0.1*feat_tot),
                                      replace=False)
        train_index = np.array(list(set(range(feat_tot)) - set(test_index)))
        test_features = features[test_index]
        features = features[train_index]
        np.savetxt('x_test.txt', test_features, fmt='%.16f')
        np.savetxt(os.path.join(config["model_save_path"],
                                'test_index.txt'),
                   test_index, fmt='%.0f')

    np.savetxt('x_data_init.txt',features, fmt='%.16f')

    if config['clean_by_pearson_r']:
        hfp = HecFeatureParser('x_data_init.txt')
        hfp.clean_by_pearson_r(save_file=True)
    if config['reduce_dimension_by_pca']:
        hfp = HecFeatureParser('x_data_init.txt')
        hfp.reduce_dimension_by_PCA(n_components=0.995,
                                        save_model=save_PCA,
                                        load_model=load_PCA,
                                        model_path=os.path.join(config["model_save_path"],
                                                                'pca_model.joblib'),
                                        save_file=True)

def y_main(config:dict):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    elif isinstance(config, dict):
        pass
    else:
        raise FileTypeError('The input config should be a name of json file or a `dict`.')

    config = {
        'split_test': config['split_test']\
            if config.__contains__('split_test') else False,
        'labels': config['labels']\
            if config.__contains__('labels') else ['B', 'G', 'E', 'Hv', 'C11', 'C44'],
        'HECC_properties_path': config['HECC_properties_path']\
            if config.__contains__('HECC_properties_path') else 'HECC_properties.CSV',
             }

    hpp = HecPropertyParser(config['HECC_properties_path'])
    labels = [hpp.get_property_from_str(x, dtype=float) for x in config['labels']]
    labels = np.vstack(labels).T

    if config['split_test']:
        test_index  = np.loadtxt(os.path.join(config["model_save_path"], 'test_index.txt'),
                                 dtype=int)
        test_labels = labels[test_index]
        labels      = labels[train_index]
        np.savetxt('y_test.txt', test_labels, fmt='%.8f')
    np.savetxt('y_data.txt', labels, fmt='%.8f')

def x_soap_main(config: dict):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    elif isinstance(config, dict):
        pass
    else:
        raise FileTypeError('The input config should be a name of json file or a `dict`.')

    if config['soap_features']:
        config_soap = config["soap_config"]

        hpp      = HecPropertyParser(config['HECC_properties_path'])
        formulas = hpp.get_property_from_str('nominal_formula', dtype=str)
        fnames = [f'CONTCAR_{x}' for x in formulas]
        sf = soap_feature('CONTCARs_new_calc')
        sf.get_x_data(fnames, config_soap)

def y_soap_main(config:dict):
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    elif isinstance(config, dict):
        pass
    else:
        raise FileTypeError('The input config should be a name of json file or a `dict`.')

    # feature_order = np.loadtxt('order_of_structures.txt', dtype=str)
    # feature_order = [x.split('_')[1:] for x in feature_order]
    # feature_order = ['_'.join(x) for x in feature_order]

    hpp = HecPropertyParser(config['HECC_properties_path'])
    labels = [hpp.get_property_from_str(x, dtype=float) for x in config['labels']]
    labels = np.vstack(labels).T
    np.savetxt('y_data_soap.txt', labels, fmt='%.8f')

if __name__ == '__main__':
    x_main('input_config.json', load_PCA=False, save_PCA=True)
    y_main('input_config.json')

    # x_soap_main('input_config.json')
    # y_soap_main('input_config.json')
