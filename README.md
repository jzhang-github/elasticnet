# **Machine learning model for predicting multi-component transition metal carbides (MTMCs)**
### This is the manual to reproduce  results and support conclusions of ***Lattice Distortion Informed Exceptional Multi-Component Transition Metal Carbides Discovered by Machine Learning***.


pip icon, conda icon, package doi icon.


We recommend using a Linux/Windows operating system to run the following examples, under the [current directory](.).  


![ML-workflow](files/Figure_1.svg)

# Table of Contents
- [Installation](#Installation)  
- [Example of using the well-trained model](#example-of-using-the-well-trained-model)   
- [Train a new model from scratch](#train-a-new-model-from-scratch)   
  - [Prepare VASP calculations](#prepare-VASP-calculations)  
  - [Collect VASP results](#collect-VASP-results)  
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

- Download the well-trained parameters: [checkpoint](#checkpoint)  
- Run the following python code:  
```
from HeccLib import predict_formula  
pf = predict_formula(config='input_config.json',ckpt_file='checkpoint')  
pf.predict(*['VNbTa', 'TiNbTa'])  
```
- The mechanical properties of (VNbTa)C3 and (TiNbTa)C3 will show on the screen. The specific moduli of each column are: B, G, E, Hv, C11, C44.
```
array([[294.43195 , 203.70157 , 496.67032 ,  25.989697, 632.3356  ,
        175.50716 ],
       [283.17245 , 201.96506 , 489.7816  ,  26.824062, 607.07336 ,
        178.52579 ]], dtype=float32)
```

# Train a new model from scratch

### Prepare VASP calculations

### Collect input features and labels

### Train

### Check training results

### Predict

### High-throughput predict

### Ternary plot

