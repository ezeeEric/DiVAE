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

### Input Data

|  Dataset | Location |
| ------------- | ------------- |
| MNIST  | TODO |
| Calorimeter Data (GEANT4 showers, ⟂ to center) | [![DOI](https://zenodo.org/badge/DOI/10.17632/pvn3xc3wy5.1.svg)](https://doi.org/10.17632/pvn3xc3wy5.1)|

MNIST dataset.

Calorimeter Image dataset is based on work in [2].

## Setup
```
git clone git@github.com:ezeeEric/DiVAE.git
cd DiVAE
```
### Installation
```

```

### After Installation
```
source source.me
```
Sources the virtual environment and appends to `PYTHONPATH`.

## How To...
### ...run models
```
python scripts/rundiVAE.py -c configs/default_divae_mnist.cfg 
```

## Notes
- In Hierarchical Encoder: probabilities are clamped (clipped)
more clamping in KLD

### References
[1] Jason Rolfe, Discrete Variational Autoencoders,
http://arxiv.org/abs/1609.02200
[2] M. Paganini ([@mickypaganini](https://github.com/mickypaganini)), L. de Oliveira ([@lukedeo](https://github.com/lukedeo)), B. Nachman ([@bnachman](https://github.com/bnachman)), _CaloGAN: Simulating 3D High Energy Particle Showers in Multi-Layer Electromagnetic Calorimeters with Generative Adversarial Networks_ [[`arXiv:1705.02355`](https://arxiv.org/abs/1705.02355)].