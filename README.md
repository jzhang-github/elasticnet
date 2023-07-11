# **Machine learning model for predicting multi-component transition metal carbides (MTMCs)**
### This is the manual to reproduce  results and support conclusions of ***Lattice Distortion Informed Exceptional Multi-Component Transition Metal Carbides Discovered by Machine Learning***.

We recommend using a Linux/Windows operating system to run the following examples, under the [current directory](.).  


![ML-workflow](files/Figure_1.svg)

# Table of Contents
- [Installation](#Installation)  
- [Example of using the well-trained model](#example-of-using-the-well-trained-model)   
- [Train a new model from scratch](#train a new model from scratch)   
  - [Prepare VASP calculations](#prepare-VASP-calculations)  
  - [Collect VASP results](#collect-VASP-results)  
  - [Collect input features and labels](#collect-input-features-and-labels)  
  - [Train](#train)  
  - [Check training results](#check-training-results)   
  - [Predict](#predict)  
  - [High-throughput predict](#high-throughput-predict)  

- [Other scripts](#other-scripts)
  - [Figure 2](#figure-2)
  - [Figure 3](#figure-3)
  - [Figure 4](#figure-4)
  - [Figure 5](#figure-5)

# Installation
**Requirements file:** [requirements.txt](requirements.txt)

**Key modules**
```
dgl-cu110 (dgl.__version__ = '0.6.1')  
numpy==1.19.5
scikit-learn==0.24.2
tensorflow==2.5.0
tensorflow-gpu==2.4.0
ase==3.21.1
pymatgen==2020.11.11
```

# Example of using the well-trained model

# Train a new model from scratch

### Prepare VASP calculations

### Collect input features and labels

### Train

### Check training results

### Predict

### High-throughput predict
