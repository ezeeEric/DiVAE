
"""
Variational Autoencoder

Main Module

Author: Eric Drechsler (eric_drechsler@sfu.ca)

Based on work from Olivia di Matteo.
"""

#pyTorch: Open source ML library dev. mainly by Facebook's AI research lab
import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np

from torch.distributions import Normal, Uniform, Bernoulli
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(1)

class VAE(nn.Module):
    def __init__(self, 
            latent_dimensions=32,   
            rbm_block_size=16,
            smoother=None
            ):
        super(VAE, self).__init__()

        self._encoderNodes=[(784,128),]

        self._reparamNodes=(128,latent_dimensions)  

        self._decoderNodes=[(latent_dimensions,128),]

        self._outputNodes=(128,784)     

        self._encoderLayers=nn.ModuleList([])
        self._decoderLayers=nn.ModuleList([])
        self._reparamLayers=nn.ModuleDict({'mu':nn.Linear(self._reparamNodes[0],self._reparamNodes[1]),
                             'var':nn.Linear(self._reparamNodes[0],self._reparamNodes[1])
        })
        self._outputLayer=nn.Linear(self._outputNodes[0],self._outputNodes[1])

        for node in self._encoderNodes:
          self._encoderLayers.append(
              nn.Linear(node[0],node[1])
              )
        for node in self._decoderNodes:
          self._decoderLayers.append(
              nn.Linear(node[0],node[1])
              )        

        # activation functions per layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #smoothing function used to 'blur' discrete latent variables
        # self.smoother = smoother 

        ###################################################################################
        ###################################### PRIOR ######################################
        ###################################################################################

        # The last set of parameters we need to worry about are those of the RBM prior.
        # Let's initialize these to some random values, and update them as we go along
        self.rbm_block_size = rbm_block_size

        self.rbm_weights = nn.Parameter(Normal(loc=0, scale=0.01).sample((self.rbm_block_size, self.rbm_block_size)))
        
        # Should use the proportion of training vectors in which unit i is turned on
        # For now let's just set them randomly 
        self.rbm_z1_bias = nn.Parameter(Uniform(low=0, high=1).sample((self.rbm_block_size, )))

        # Unless there is some sparsity, initialize these all to 0
        #self._hidden_bias = nn.Parameter(torch.zeros((self.n_hidden, )))
        self.rbm_z2_bias = nn.Parameter(Uniform(low=-0.1, high=0.1).sample((self.rbm_block_size, )))
    
    def encode(self, x):
        for layer in self._encoderLayers:
          x=self.relu(layer(x))
        # Split for reparameterization
        mu = self._reparamLayers['mu'](x)
        logvar = self._reparamLayers['var'](x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """ Sample from the normal distributions corres and return var * samples + mu
        """
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5 * logvar)
        
    def decode(self, z):
        for layer in self._decoderLayers:
          z=self.relu(layer(z))
        x_prime = self.sigmoid(self._outputLayer(z))                 
        return x_prime
                            
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_prime = self.decode(z)
        return x_prime, mu, logvar    
    
    def print_model_info(self):
        for par in self.__dict__.items():
            logger.debug(par)

if __name__=="__main__":
    print("Testing Model Setup for VAE")
    model=VAE()
    print("Success")
    pass