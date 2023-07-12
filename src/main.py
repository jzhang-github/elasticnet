# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:55:04 2023

@author: ZHANG Jun
"""
def main():
    # prepare dataset
    from prepare_input import x_main, y_main
    x_main('input_config.json', load_PCA=False, save_PCA=True)
    y_main('input_config.json')

    # train
    from ANN import CV_ML_RUN, load_and_pred
    CV_ML_RUN('train.json')
    load_and_pred('train.json', 'x_data_after_pca.txt', write_pred_log=True, drop_cols=None)
