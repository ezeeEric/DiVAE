#DiVAE [ˈdiːvə]

A package containing neural network architectures based on Variational
Autoencoders (VAE) and Restricted Boltzmann Machines (RBM).
Implemented are:
- Vanilla Autoencoder
- Vanilla VAE
- conditional VAE
- sequential VAE to process successive images (i.e. calorimeter data)
- VAE with hierarchical posterior
- Discrete VAE [1]
- Vanilla RBM
- ...

## Overview
### Repository Structure


| Directory        | Content    | 
| ------------- |:-------------| 
| `configs/`      | Configuration files | 
| `models/` | Core module, includes definitions of all models.  |
| `sandbox/` | Collection of test scripts and standalone models. |
| `scripts/` | Steering scripts. |
| `utils/` | Helper functionalities for core modules (plotting etc.) |
| `data/` | Location of input data after download |
| `output/` | Output Location |

### Input Data

|  Dataset | Location |
| ------------- | ------------- |
| MNIST  | retrieved through torchvision |
| Calorimeter Data (GEANT4 showers, ⟂ to center) | [![DOI](https://zenodo.org/badge/DOI/10.17632/pvn3xc3wy5.1.svg)](https://doi.org/10.17632/pvn3xc3wy5.1)|

data path:


## Setup
```
git clone git@github.com:ezeeEric/DiVAE.git
cd DiVAE
git checkout -b mydev
```

### Installation
#### Via Virtual Environment and pip
Initial package setup:
```
python3 -m venv venv_divae
source source.me
python3 -m pip install -r requirements.txt
```

### After Installation
After the initial setup, simply navigate to the package directory and run

```
source source.me
```
Sources the virtual environment and appends to `PYTHONPATH`.

### Data Path


## How To...
### ...run models
```
python scripts/rundiVAE.py -c configs/default_divae_mnist.cfg 
```

## Notes
- In Hierarchical Encoder: probabilities are clamped (clipped)
more clamping in KLD

- The configuration object config is instantiated in the __init__ file of the
  main module and imported as global variable.

### References
[1] Jason Rolfe, Discrete Variational Autoencoders,
http://arxiv.org/abs/1609.02200

[2] M. Paganini ([@mickypaganini](https://github.com/mickypaganini)), L. de Oliveira ([@lukedeo](https://github.com/lukedeo)), B. Nachman ([@bnachman](https://github.com/bnachman)), _CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks_ [[`arXiv:1705.02355`](https://arxiv.org/abs/1705.02355)].