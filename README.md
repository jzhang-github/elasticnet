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

# Installation

### Install under [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment

- Create a new environment   
```
conda create -n MTMC python==3.10
```

- Activate the environment  
```
conda activate MTMC
```

- Install package  
```
conda install mtmc
```

### Alternatively, you can install with [pip](https://pypi-url).
- Install the package  
```
pip install mtmc
```

- **If your IP locates in mainland China, you may need to install it from the tsinghua mirror.**  
```
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
```
from HeccLib import predict_formula  
pf = predict_formula(config='input_config.json',ckpt_file='checkpoint')  
pf.predict(*['VNbTa', 'TiNbTa'])  
```
- The mechanical properties of (VNbTa)C3 and (TiNbTa)C3 will show on the screen. The specific modulus of each column is: B, G, E, Hv, C11, C44.
```
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
- 


### Check training results

### Predict

### High-throughput predict

### Ternary plot

