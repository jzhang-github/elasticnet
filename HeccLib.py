# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:43:47 2022

@author: ZHANG Jun
"""

import numpy as np
from ase.formula import Formula
import json
import pandas as pd
import os
from ase.io import read, write

class FileTypeError(Exception):
    pass

def get_number_of_components(f):
    formula = Formula(f).count()
    effective_components = [x for x in formula if formula[x] != 0]
    real_formula = ''.join([f'{x}{formula[x]}' for x in formula if formula[x] != 0])
    return len(effective_components), real_formula

class HecFeatureGenerator(object):
    def __init__(self,
                 prop_precursor_path='properties_of_precursors.json',
                 props = ['V', 'm', 'rho', 'VEC', 'X',
                          'reduced_lattice_constant', 'group', 'periodic'],
                 operators=[np.sqrt, np.log10, np.log, np.square, 'cube', 'plus', 'minus', 'multiply']):
        self._prop_precursor_path = prop_precursor_path

        if prop_precursor_path.split('.')[1] == 'json':
            self.pre_prop = self.get_precursor_properties_from_json()
        elif prop_precursor_path.split('.')[1] == 'csv' or prop_precursor_path.split('.')[1] == 'CSV':
            self.pre_prop = self.get_precursor_properties_from_csv()
        else:
            raise FileTypeError

        self.props = props

        def plus(feat):
            feat = np.array(feat)
            col = np.shape(feat)[1]
            new_feat = []
            for i in range(col):
                for j in range(i+1, col):
                    new_feat.append(feat[:,i] + feat[:,j])
            return np.array(new_feat).T

        def minus(feat):
            feat = np.array(feat)
            col = np.shape(feat)[1]
            new_feat = []
            for i in range(col):
                for j in range(i+1, col):
                    new_feat.append(feat[:,i] - feat[:,j])
            return np.array(new_feat).T

        def multiply(feat):
            feat = np.array(feat)
            col = np.shape(feat)[1]
            new_feat = []
            for i in range(col):
                for j in range(i+1, col):
                    new_feat.append(feat[:,i] * feat[:,j])
            return np.array(new_feat).T

        self.func = {'cube': lambda feat: np.power(feat, 3),
                     'exp_n': lambda feat: 1/np.exp(np.clip(feat, a_min=None, a_max=200)), # `np.clip` is used to avoid the zero-division error
                     'exp': lambda feat: np.exp(np.clip(feat, a_min=None, a_max=200)),
                     'plus': plus,
                     'minus': minus,
                     'multiply': multiply,
                     'sqrt': np.sqrt,
                     'log10': np.log10,
                     'log': np.log,
                     'square': np.square
                     }

        self.operators=[]
        for op in operators:
            if isinstance(op, str):
                self.operators.append(self.func[op])
            else:
                self.operators.append(op)

    def get_precursor_properties_from_json(self):
        with open(self._prop_precursor_path, 'r') as json_file:
            pre_prop = json.load(json_file)
        return pre_prop

    def get_precursor_properties_from_csv(self):
        df = pd.read_csv(self._prop_precursor_path)
        # df = df.iloc[:,1:]
        df_json = {}
        for j, k in enumerate(df.iloc[:,0]):
            df_json[k] = {}
            for ki in df.columns.values.tolist():
                df_json[k][ki] = df[ki][j]
        return df_json

    def get_concentration_from_ase(self, *formula: str): # for carbides only
        # the C element will be depressed in this function.
        con = []
        # print(formula)
        for f in formula:
            f_dict = Formula(f).count()
            if f_dict.__contains__('C'):
                del f_dict['C']
            tot = np.sum(list(f_dict.values()))
            c_dict = {k+'C': v/tot for k, v in f_dict.items() if v != 0.0}
            con.append(c_dict)
        return con

    def get_concentration_from_ase_new(self, *formula): # for alloys
        # all elements in ase formula will be counted.
        con = []
        for f in formula:
            f_dict = Formula(f).count()
            tot = np.sum(list(f_dict.values()))
            c_dict = {k: v/tot for k, v in f_dict.items()}
            con.append(c_dict)
        return con

    def get_concentration_from_pymatgen(self, *formula):
        pass

    def get_mixing_entropy(self, *ele_con_dict): # ele_con_dict: a list of dict.
        entropy = []
        for ele_con in ele_con_dict:
            e = 0.0
            for c in ele_con.values():
                e += -8.3145 * c * np.log(c)
            entropy.append(e)
        return np.array(entropy)

    def get_mixing_enthalpy(self, energy_list, ele_con_dict):
        assert len(energy_list) == len(ele_con_dict), f'energy_list and ele_con\
_dict are incompatible with length: energy_list: {len(energy_list)}ele_con_dict:\
{len(ele_con_dict)}'
        num = len(energy_list)
        enthalpy_list = []
        for i in range(num):
            enthalpy = energy_list[i]
            for pre in ele_con_dict[i]:
                enthalpy -= self.pre_prop[pre]['total_energy_per_formula'] *\
                    ele_con_dict[i][pre]
            enthalpy_list.append(enthalpy)
        return enthalpy_list

    def get_volume_change(self, volume_list, ele_con_dict):
        assert len(volume_list) == len(ele_con_dict), f'energy_list and ele_con\
_dict are incompatible with length: energy_list: {len(volume_list)}ele_con_dict:\
{len(ele_con_dict)}'
        num = len(volume_list)
        volume_diff_list = []
        for i in range(num):
            v_diff = volume_list[i]
            for pre in ele_con_dict[i]:
                v_diff -= self.pre_prop[pre]['volume_per_formula'] *ele_con_dict[i][pre]
            volume_diff_list.append(v_diff)
        return volume_diff_list

    def get_ave_std_span_prop(self,
                              *ele_con_dict):
        col_num = len(self.props)
        row_num = len(ele_con_dict)
        ave, std, span = np.zeros(shape=(row_num, col_num)), np.zeros(shape=(row_num, col_num)), np.zeros(shape=(row_num, col_num))

        for i, ele_con in enumerate(ele_con_dict):
            for j, prop in enumerate(self.props):
                tmp_list = [self.pre_prop[x][prop] for x in ele_con]
                con_list = list(ele_con.values())
                num = len(tmp_list)
                ave[i,j] = np.sum(np.array(tmp_list) * np.array(con_list)) + 1e-14 # 1e-14 is used for the np.log operator
                tmp = [(1 - tmp_list[x] / ave[i,j]) ** 2 for x in range(num)]
                tmp = [con_list[i] * x for i, x in enumerate(tmp)]
                std[i,j] = np.sqrt(np.sum(tmp)) + 1e-14 # 1e-14 is used for the np.log operator
                span[i,j] = np.max(tmp_list) - np.min(tmp_list) + 1e-14 # 1e-14 is used for the np.log operator
        return ave, std, span

    def get_ave_from_formula(self, props, *formulas):
        # assert prop in self.prop, 'Input prop is not parsered.'
        if isinstance(props, str):
            col_num = 1
            props = [props]
        elif isinstance(props, list):
            col_num = len(props)

        ele_con_dict = self.get_concentration_from_ase(*formulas)

        row_num = len(ele_con_dict)
        ave = np.zeros(shape=(row_num, col_num))

        for i, ele_con in enumerate(ele_con_dict):
            for j, prop in enumerate(props):
                tmp_list = [self.pre_prop[x][prop] for x in ele_con]
                con_list = list(ele_con.values())
                ave[i,j] = np.sum(np.array(tmp_list) * np.array(con_list)) + 1e-14 # 1e-14 is used for the np.log operator
        return ave

    def expand_features(self, feat):
        new_feat=[]
        for op in self.operators:
            new_feat.append(op(feat))
        new_feat = np.hstack(new_feat)
        return new_feat

    def scale_features(self, feat, scale_factor=1.3,
                       save_log=True,
                       load_log=False,
                       log_file=os.path.join('checkpoint',
                                             'scale_range.json')):
        if load_log:
            with open(log_file, 'r') as json_file:
                j_out = json.load(json_file)
            max_array = np.array(j_out['max'])
            min_array = np.array(j_out['min'])
            scale_factor = j_out['scale_factor']
        else:
            max_array = np.max(feat, axis=0)
            min_array = np.min(feat, axis=0)
        span_array = max_array - min_array
        # print(max_array, min_array, span_array)
        new_feat = (feat - min_array + 0.5 * span_array * (scale_factor - 1.0)) / (span_array * scale_factor + 0.01) # 0.01 is used to aovid zero-division error.

        if save_log:
            if not os.path.exists('checkpoint'):
                os.mkdir('checkpoint')
            with open(log_file, 'w') as json_file:
                json.dump({'max': max_array.tolist(),
                           'min': min_array.tolist(),
                           'scale_factor': scale_factor}, json_file, indent=4)
        return new_feat

    def json2csv_test(self, in_name, out_name):
        with open('in_name', 'r') as json_file:
            j_out = json.load(json_file)
        out = []
        for i,f in enumerate(j_out):
            out.append([])
            for j in j_out[f]:
                out[i].append(j_out[f][j])
        test = np.array(out)
        np.savetxt(out_name, test, fmt='%.16f')

    def get_input_feature_from_formula(self, *formula, scale_factor=1.3,
                                       save_log=True,
                                       load_log=False,
                                       log_file=os.path.join('checkpoint',
                                                             'scale_range.json')):
        con = self.get_concentration_from_ase(*formula)
        mixing_entropy = self.get_mixing_entropy(*con) # / 14 # the maximum entropy of HEC with 5 metallics are 13.3816715.
        ave, std, span = self.get_ave_std_span_prop(*con)
        features = np.hstack([np.reshape(mixing_entropy, (-1, 1)), ave, std, span])
        features = self.scale_features(features, scale_factor, save_log, load_log, log_file)
        if load_log:
            features = np.where(features < 1e-14, 1e-14, features)
        features_expand = self.expand_features(features)
        features_expand = self.scale_features(features_expand, scale_factor,
                                              save_log, load_log,
                                              os.path.join('checkpoint',
                                                           'scale_range_1.json'))
        features = np.hstack([features, features_expand])
        return features

    def get_input_feature_from_formula_new(self, *formula, scale_factor=1.3,
                                       save_log=True,
                                       load_log=False,
                                       log_file='scale_range.json'):
        con = self.get_concentration_from_ase(*formula)
        mixing_entropy = self.get_mixing_entropy(*con) # / 14 # the maximum entropy of HEC with 5 metallics are 13.3816715.
        ave, std, span = self.get_ave_std_span_prop(*con)
        ave = np.hstack([np.reshape(mixing_entropy, (-1, 1)), ave])
        ave_expand = self.expand_features(ave)
        features = np.hstack([ave, std, span, ave_expand, ave_expand])
        features = self.scale_features(features, scale_factor, save_log, load_log, log_file)
        return features

    def get_input_feature_from_real_formula(self, *formula, scale_factor=1.3,
                                       save_log=True, file=None):
        con = self.get_concentration_from_ase_new(*formula)
        mixing_entropy = self.get_mixing_entropy(*con) # / 14 # the maximum entropy of HEC with 5 metallics are 13.3816715.
        ave, std, span = self.get_ave_std_span_prop(*con)
        features = np.hstack([np.reshape(mixing_entropy, (-1, 1)), ave, std, span])
        features = self.scale_features(features, scale_factor, save_log, file)
        return features

class real_mae(object):
    def __init__(self, fname='label_range.json'):
        import tensorflow as tf
        self.fname = fname
        with open(self.fname, 'r') as f:
            label_config = json.load(f)
        self.y_max = label_config['y_max']
        self.y_min = label_config['y_min']

    def update(self, y_true, y_pred):
        y_real_pred = y_pred * (self.y_max - self.y_min) + self.y_min
        y_real_true = y_true * (self.y_max - self.y_min) + self.y_min
        mae = tf.keras.losses.MeanAbsoluteError(y_real_true, y_real_pred)
        return mae

class fraction2real(object):
    def __init__(self, fname='label_range.json'):
        import tensorflow as tf
        self.fname = fname
        with open(self.fname, 'r') as f:
            label_config = json.load(f)
        self.y_max = label_config['y_max']
        self.y_min = label_config['y_min']

    def update(self, y_true, y_pred):
        y_real_pred = y_pred * (self.y_max - self.y_min) + self.y_min
        y_real_true = y_true * (self.y_max - self.y_min) + self.y_min
        return y_real_pred, y_real_true

    def mae(self, y_true, y_pred):
        return tf.keras.losses.MeanAbsoluteError(y_true, y_pred)

class HecPropertyParser(object): # label parser
    def __init__(self, fname='HEC_properties.txt'):
        self.fname = fname
        if fname.split('.')[1] == 'txt':
            self.data = np.loadtxt(self.fname, dtype=str)
        elif fname.split('.')[1] == 'csv' or fname.split('.')[1] == 'CSV':
            df = pd.read_csv(self.fname)
            prop_name = list(df)
            data = np.array([df[x].to_numpy() for x in prop_name]).T
            data = np.vstack((np.reshape(prop_name, (1,-1)), data))
            self.data = data
        else:
            raise FileTypeError

    def get_property_from_str(self, prop, dtype=str):
        items = self.data[0]
        assert prop in items, f'{prop} cannot be found in {items}'
        col = np.where(items==prop)
        prop_data = self.data[1:, col].reshape(-1)
        return prop_data.astype(dtype)

    def get_property_from_str_based_on_formula(self, prop, formulas, dtype=str):
        items = self.data[0]
        assert prop in items, f'{prop} cannot be found in {items}.'
        assert 'nominal_formula' in items, 'nominal_formula cannot be found in {items}.'

        f_col = np.where(items=='nominal_formula')
        nominal_formulas = self.data[1:, f_col].reshape(-1)

        col = np.where(items==prop)
        prop_data = self.data[1:, col].reshape(-1)
        prop_dict = dict(zip(nominal_formulas, prop_data))

        prop_order = np.array([prop_dict[x] for x in formulas])
        return prop_order.astype(dtype)

    def save_properties_as_dict(self, props=[]):
        pass

    # def scale_labels(self):
    #     pass

class HecFeatureParser(object):
    def __init__(self, fname='x_data.txt'):
        self.fname = fname

        if isinstance(self.fname, np.ndarray):
            self.data = self.fname
        elif isinstance(self.fname, str):
            self.data  = np.loadtxt(self.fname, dtype=float)
        else:
            raise FileTypeError('The input should be a name of txt file or a `np.array`.')

    def get_correlation_matrix(self, r_cutoff=0.95, save_file=True):
        # output index starts from 1.
        R = np.corrcoef(self.data.T)
        row, col = np.where(R > r_cutoff)
        diag = np.where(row == col)
        row, col = np.delete(row, diag)+1, np.delete(col, diag)+1
        redundant1 = list(set(row[np.where(row < col)]))
        redundant2 = list(set(row[np.where(row > col)]))
        redundant = [redundant1, redundant2][np.argmin([len(redundant1), len(redundant2)])]
        num = len(R)
        R = np.vstack((np.array(range(1, num+1)), R))
        R = np.hstack((np.array(range(0, num+1)).reshape((-1,1)),
                       R))
        if save_file:
            np.savetxt('correlation_matrix.txt', R, fmt='%.4f')
        return R, redundant

    def clean_by_pearson_r(self, save_file=True):
        R, redundant = self.get_correlation_matrix(save_file=save_file)
        data_new = self.data.copy()
        data_new = np.delete(data_new, [x-1 for x in redundant], axis=1)
        print('Input shape after removing:', np.shape(data_new))
        if save_file:
            np.savetxt('x_data_after_pr.txt', data_new, fmt='%.16f')
            if not os.path.exists('checkpoint'):
                os.mkdir('checkpoint')
            np.savetxt(os.path.join('checkpoint', 'removed_feature_index.txt'),
                       [x-1 for x in redundant], fmt='%.0f')
        return data_new

    def reduce_dimension_by_PCA(self,
                                n_components: float=0.97,
                                save_model=True,
                                load_model=False,
                                model_path=os.path.join('checkpoint',
                                                        'pca_model.joblib'),
                                save_file=True):
        from sklearn.decomposition import PCA
        import joblib
        if load_model:
            pca_f = joblib.load(model_path)
        else:
            pca_f = PCA(n_components=n_components, svd_solver='full')
            pca_f.fit(self.data)
        data_new = pca_f.transform(self.data)
        if save_model:
            if not os.path.exists('checkpoint'):
                os.mkdir('checkpoint')
            joblib.dump(pca_f, model_path)
        if save_file:
            np.savetxt('x_data_after_pca.txt', data_new, fmt='%.16f')
        print(pca_f.explained_variance_)
        print(pca_f.explained_variance_ratio_)
        print(pca_f.explained_variance_ratio_.sum())
        print(pca_f.singular_values_)
        print('Input shape after PCA:', np.shape(data_new))
        return data_new

    def get_range(self):
        return self.get_max() - self.get_min()

    def get_max(self):
        return np.max(self.data, axis=0)

    def get_min(self):
        return np.min(self.data, axis=0)

class predict_formula(object):
    # for now, use PCA method only.
    def __init__(self, config='input_config.json', ckpt_file='checkpoint'):
        import joblib
        if isinstance(config, dict):
            self.config = config
        elif isinstance(config, str):
            with open('input_config.json') as f:
                self.config = json.load(f)
        self.checkpoint = ckpt_file

        self.models = self.get_models()
        self.hfg = HecFeatureGenerator(prop_precursor_path=self.config['prop_precursor_path'],
                                  props=self.config['props'],
                                  operators=self.config['operators'])
        self.pca_model = joblib.load(os.path.join(self.checkpoint,
                                                  'pca_model.joblib'))

    def get_models(self):
        import tensorflow as tf
        # import model
        model_names = os.listdir(os.path.join(self.checkpoint, 'cp.ckpt'))
        models = []
        for model_name in model_names:
            model_name = os.path.join(self.checkpoint, 'cp.ckpt', model_name)
            model = tf.keras.models.load_model(model_name)
            models.append(model)
        return models

    def predict(self, *formulas):
        feat_init = self.hfg.get_input_feature_from_formula(*formulas,
                                                       save_log=False,
                                                       load_log=True)
        # pass into pca. By default, the input dimensions are reduced by PCA.
        feat = self.pca_model.transform(feat_init)

        predictions_all = []
        for model in self.models:
            predictions = model.predict([feat])
            predictions_all.append(predictions)

        predictions_all = np.array(predictions_all)
        prediction_mean = np.mean(predictions_all, axis=0)
        predictions_all = np.concatenate(predictions_all, axis=1)
        return prediction_mean

def get_concentration_from_ase(*formula: str): # for carbides only
    # the C element will be depressed in this function.
    con = []
    for f in formula:
        f_dict = Formula(f).count()
        if f_dict.__contains__('C'):
            del f_dict['C']
        tot = np.sum(list(f_dict.values()))
        c_dict = {k: v/tot for k, v in f_dict.items() if v != 0.0}
        con.append(c_dict)
    return con

def get_models(path='checkpoint'):
    import tensorflow as tf
    # import model
    model_names = os.listdir(os.path.join(path, 'cp.ckpt'))
    models = []
    for model_name in model_names:
        model_name = os.path.join(path, 'cp.ckpt', model_name)
        model = tf.keras.models.load_model(model_name)
        models.append(model)
    return models

class soap_feature(object):
    def __init__(self, structure_path = 'CONTCARs'):
        # self.structure_mode = structure_mode
        self.structure_path = structure_path

    def soap_pca(self,
                 input_data,
                 n_components: float=0.97,
                 save_model=True,
                 load_model=False,
                 model_path=os.path.join('checkpoint',
                                    'soap_pca_model.joblib'),
                 save_file=True):
        from sklearn.decomposition import PCA
        import joblib
        if load_model:
            pca_f = joblib.load(model_path)
        else:
            pca_f = PCA(n_components=n_components, svd_solver='full')
            pca_f.fit(input_data)
        data_new = pca_f.transform(input_data)
        if save_model:
            if not os.path.exists('checkpoint'):
                os.mkdir('checkpoint')
            joblib.dump(pca_f, model_path)
        if save_file:
            np.savetxt('x_data_soap_after_pca.txt', data_new, fmt='%.16f')
        print(pca_f.explained_variance_ratio_.sum())
        print('Input shape after PCA:', np.shape(data_new))
        return data_new

    def get_soap_feature(self, ase_atoms, nmax=5, lmax=5, n_jobs=1,sigma=0.5, rcut=5.0,
                         species=['C', 'Ti', 'V', 'Cr', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W']):
        from dscribe.descriptors import SOAP
        soap = SOAP(
            species=species,
            periodic=True,
            rcut=rcut,
            nmax=nmax,
            lmax=lmax,
            sigma=sigma,
            average="inner",
            sparse=False)
        x_tmp = soap.create(ase_atoms,n_jobs=n_jobs)
        return x_tmp

    def get_contcar_atoms(self, fnames:list, path2structure='CONTCARs'):
        if fnames:
            fnames = fnames
        else:
            fnames = os.listdir(path2structure)
        return [read(os.path.join(path2structure, x)) for x in fnames]

    def shift_atoms1_to_atoms2(self, atoms1, atoms2, cutoff=1.0): # atoms2 should be the reference structure.
        cell = atoms2.get_cell()
        atoms1.set_cell(cell, scale_atoms=True)
        pos_ref = atoms2.get_positions()
        pos_1   = atoms1.get_positions()
        pos_diff = pos_1[0] - pos_ref[0]
        pos_1 -= pos_diff
        atoms1.set_positions(pos_1)
        atoms1.wrap()

        for atom in atoms1:
            atoms_tmp = atoms2.copy()
            atoms_tmp.append(atom)
            dists = atoms_tmp.get_distances(-1, range(len(atoms_tmp)-1), mic=True)
            # print(dists)
            site = np.where(dists < cutoff)[0][0]
            # print(site)
            atom.position = pos_ref[site]
        return atoms1

    def get_poscar_atoms(self, fnames:list, path2structure='CONTCARs', mode='ref', reference_file='POSCAR_ref'):
        if fnames:
            fnames = fnames
        else:
            fnames = os.listdir(path2structure)
        if mode == 'file':
            print('Read structure from POSCAR directly.')
            return [read(os.path.join(path2structure, x)) for x in fnames]
        elif mode == 'ref':
            print('Read structure from CONTCAR, but shift the structure to the reference POSCAR.')
            atoms_ref = read(reference_file)
            # print('HERE', fnames)
            return [self.shift_atoms1_to_atoms2(read(os.path.join(path2structure, x)), atoms_ref) for x in fnames]

    def get_x_data(self, fnames:list, soap_config='soap_config.json'): # input_structure_type='POSCAR',
        if isinstance(soap_config, str):
            with open(soap_config, 'r') as f:
                soap_config = json.load(f)

        if soap_config['input_structure_type'] == 'POSCAR':
            atoms_list = self.get_poscar_atoms(fnames, self.structure_path)
        elif soap_config['input_structure_type'] == 'CONTCAR':
            atoms_list = self.get_contcar_atoms(fnames, self.structure_path)

        x_raw = self.get_soap_feature(atoms_list,
                                      nmax=soap_config['nmax'],
                                      lmax=soap_config['lmax'],
                                      rcut=soap_config['rcut'],
                                      n_jobs=soap_config['n_jobs'],
                                      sigma=soap_config['sigma'],
                                      species=soap_config['species'])

        x_after_pca = self.soap_pca(x_raw,
                                    n_components=soap_config["n_components"],
                                    save_model=True,
                                    load_model=False,
                                    model_path=os.path.join('checkpoint',
                                                            'soap_pca_model.joblib'),
                                    save_file=True)

# # debug
# with open('input_config.json', 'r') as f:
# #     soap_config = json.load(f)['soap_config']
# for s in species:
#     atoms_tmp = read(f'CONTCAR_{s}')
#     atoms_tmp = atoms_tmp.repeat(2)
#     write(f'CONTCAR_{s}', atoms_tmp)
