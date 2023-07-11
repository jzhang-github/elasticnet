pip icon, conda icon, package doi icon.
# **Machine learning model for predicting multi-component transition metal carbides (MTMCs)**
### This is the manual to reproduce  results and support conclusions of ***Lattice Distortion Informed Exceptional Multi-Component Transition Metal Carbides Discovered by Machine Learning***.



We recommend using a Linux/Windows operating system to run the following examples, under the [current directory](.).  


![ML-workflow](files/Figure_1.svg)

# Table of Contents
- [Installation](#Installation)  
- [Example of using the well-trained model](#example-of-using-the-well-trained-model)   
- [Train a new model from scratch](#train-a-new-model-from-scratch)   
  - [Prepare DFT calculations](#prepare-DFT-calculations)  
  - [Collect DFT results](#collect-DFT-results)  
  - [Collect input features and labels](#collect-input-features-and-labels)  
  - [Train](#train)  
  - [Check training results](#check-training-results)   
  - [Predict](#predict)  
  - [High-throughput predict](#high-throughput-predict)  
  - [Ternary plot](#ternary-plot)  

- [Other scripts](#other-scripts)
  - [get rom](#get-rom)
  - [get vec](#get-vec)
  - [analysis](#figure-4)
  - [Figure 5](#figure-5)

- [Abbreviations](abbreviations)

# Installation

### Install under [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment

- Create a new environment   
```console
conda create -n MTMC python==3.10
```

- Activate the environment  
```console
conda activate MTMC
```

- Install package  
```console
conda install mtmc
```

### Alternatively, you can install with [pip](https://pypi-url).
- Install the package  
```console
pip install mtmc
```

- **If your IP locates in mainland China, you may need to install it from the tsinghua mirror.**  
```console
pip install mtmc -i https://pypi.tuna.tsinghua.edu.cn/simple
```


**Requirements file:** [requirements.txt](requirements.txt)

**Key modules**  
```
numpy==1.25.0    
scikit-learn==1.2.2   
tensorflow==2.10.0   
ase==3.22.1  
pandas==1.5.3
```

# Example of using the well-trained model  

- Download the well-trained parameters: [checkpoint](checkpoint)  
- Run the following python code:  
```python
from HeccLib import predict_formula  
pf = predict_formula(config='input_config.json',ckpt_file='checkpoint')  
pf.predict(*['VNbTa', 'TiNbTa'])  
```
- The mechanical properties of (VNbTa)C3 and (TiNbTa)C3 will show on the screen. The specific modulus of each column is: B, G, E, Hv, C11, C44.
```python
array([[294.43195 , 203.70157 , 496.67032 ,  25.989697, 632.3356  ,
        175.50716 ],
       [283.17245 , 201.96506 , 489.7816  ,  26.824062, 607.07336 ,
        178.52579 ]], dtype=float32)
```

# Train a new model from scratch
### Prepare DFT calculations
- Bulk optimization.
- Elastic constants calculation.

### Collect DFT results
- Collect elastic constants into a file with `csv` extension. See example: [files/HECC_properties_over_sample.CSV](files/HECC_properties_over_sample.CSV).  
- You may refer to these papers to calculate modulus from C11, C12, and C44: [PHYSICAL REVIEW B 87, 094114 (2013)](https://doi.org/10.1103/PhysRevB.87.094114) and [Journal of the European Ceramic Society 41 (2021) 6267-6274](https://doi.org/10.1016/j.jeurceramsoc.2021.05.022)  
- The `*csv` file should contain at least these columns: `nominal_formula`, `C11`, `C12`, `C44`, `B`, `G`, `E`, `Hv`, and `real_formula`. See example: [files/HECC_properties_over_sample.CSV](files/HECC_properties_over_sample.CSV). 

### Prepare configurations files  
- [`input_config.json`](input_config.json): Define how to generate input features and labels. You are recommended to download this file and modify then.  
  |  Variable             | Type   | Meaning                                                                                                       |  
  |  -------------------- | ----   | ------------------------------------------------------------------------------------------------------------  |  
  | include_more          | `bool` | If `True`, the `bulk_energy_per_formula` and `volume_per_formula` are also be included in the input features. |
  | split_test            | `bool` | If `True`, a new test set will be split from the dataset. For cross validation, it is OK to set this as `False`. |  
  | clean_by_pearson_r    | `bool` | Clean input features. Highly correlated features will be removed if this is `True`. |  
  | reduce_dimension_by_pca | `bool` | Clean input features by [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn-decomposition-pca). Choose one among `clean_by_pearson_r` and `reduce_dimension_by_pca`.  |
  | prop_precursor_path   | str | A file storing the properties of precursory binary carbides. File extension can be `*.csv` and `*.json`.  See example: [file/HECC_precursors.csv](file/HECC_precursors.csv)|
  | model_save_path       | str | Path for storing [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn-decomposition-pca) model and other information when generating input features and labels |  
  | props                 | list | A list of properties that are encoded into the input features. Choose among the column names of [files/HECC_precursors.csv](files/HECC_precursors.csv).  |
  | operators    | list | A list of operators to expand the input dimension. Choose among: ['cube', 'exp_n', 'exp', 'plus', 'minus', 'multiply', 'sqrt', 'log10', 'log', 'square']. | 
  | HECC_properties_path   | str | A file contains the properties of MTMCs. |

- [`train.json`](train.json): Define how to train the machine-learning model.

### Collect input features and labels  
```python    
from prepare_input import x_main, y_main
x_main('input_config.json', load_PCA=False, save_PCA=True)
y_main('input_config.json')
```

Three files will be generated:  
- `x_data_init.txt`: input features without [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn-decomposition-pca).  
- `x_data_after_pca.txt`: input features after [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn-decomposition-pca).
- `y_data.txt`: labels

### Train  
- Run the following python code on Linux OS.  
```python
from ANN import CV_ML_RUN, load_and_pred
if __name__ == '__main__':
    CV_ML_RUN('train.json')
    load_and_pred('train.json', 'x_data_after_pca.txt', write_pred_log=True, drop_cols=None)
```

- If you want to train the model on windows OS, you need to run the source code directly.  
```console
python ANN.py
```

### Check training results
- Generated files/folders  
  - `checkpoint`: 

### Predict

### High-throughput predict

### Ternary plot



# Other scripts



# Abbreviations

|  Abbr.                | Full name   |
|  -------------------- | ---- -----  |
|  MTMC                 | Multi-component transition metal carbides    |
|  HECC                 | High-entropy carbide ceramic    |
|  ML                 | Machine learning    |


