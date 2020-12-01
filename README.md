#DiVAE [ˈdiːvə]

A package containing neural network architectures based on Variational
Autoencoders (VAE) and Restricted Boltzmann Machines (RBM).
Implemented are:
- Vanilla Autoencoder
- Vanilla VAE
- conditional VAE
- sequential VAE to process successive images (i.e. calorimeter data)
- VAE with hierarchical encoder
- Discrete VAE [1]
- RBM
- ...

## Overview

### Input Data

#### MNIST

#### Calorimeter Images


## Setup
```
git clone

```
### Installation
```

```

### After Installation
```
source source.me

```

## How To...
### ...run models
```
python 

```

## Notes
- In Hierarchical Encoder: probabilities are clamped (clipped)
more clamping in KLD

### References
[1] Jason Rolfe, Discrete Variational Autoencoders,
http://arxiv.org/abs/1609.02200