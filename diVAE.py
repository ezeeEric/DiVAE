# -*- coding: utf-8 -*-
"""
Discrete Variational Autoencoder

Main Module

Author: Eric Drechsler (eric_drechsler@sfu.ca)

Based on work from Olivia di Matteo.
"""

#pyTorch: Open source ML library dev. mainly by Facebook's AI research lab
import torch
import torch.nn as nn
from torch.distributions import Normal

import numpy as np

from torch.distributions import Normal,Uniform,Bernoulli
from smoothers import SymmetricSmoother

from rbm import RBM

from networks import Encoder,Decoder,Prior

from copy import copy
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(1)

class DiVAE(nn.Module):
    def __init__(self, latent_dimensions=32):
        super(DiVAE, self).__init__()

        self.latent_dimensions=latent_dimensions
        self._encoderNodes=[(784,128),]
        self._reparamNodes=(128,latent_dimensions)  
        self._decoderNodes=[(latent_dimensions,128),]
        self._outputNodes=(128,784)     
        
        # TODO replace the above with a more elegant solution
        # self.networkStructures={
        #     'encoder':[784,128,32],
        #     'decoder':[32,128,784]
        # }

        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()
        self.prior=self._create_prior()
    
    def _create_encoder(self):
        logger.debug("_create_encoder")
        return Encoder(node_sequence=self._encoderNodes, activation_fct=nn.ReLU())

    def _create_decoder(self):
        logger.debug("_create_decoder")
        return Decoder(
            node_sequence=self._decoderNodes,
            activation_fct=nn.ReLU(),
            output_nodes=self._outputNodes,
            output_activation_fct=nn.Sigmoid(),
            )

    def _create_prior(self):
        logger.debug("_create_prior")
        return Prior(node_sequence=self._reparamNodes)

    def loss(self, x, x_recon, mu, logvar):
        logger.debug("loss")

        # Autoencoding term
        auto_loss = torch.nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
        
        #
        #encoder.hierarchical_posterior
        #decoder.reconstruct
        #createOutput
        #calculate KLD and loss

    def generate_samples(self, mu, logvar):
        logger.debug("generate_samples")
        return self.sampleZ(mu, logvar)

# pytorch forward call
# from VAE
#     def forward(self, x):
#         logger.debug("forward")
# #       encoder
# #       sample
# #       decode
#         pass
    
    def hierarchical_posterior(self,x):
        logger.debug("hierarchical_posterior")
        #dummy
        x_tilde=self.encoder.encode(x)
        mu, logvar=self.prior.reparameterize(x_tilde)
        return mu, logvar                     

    def forward(self, x):
        logger.debug("forward")
        mu, logvar = self.hierarchical_posterior(x.view(-1, 784))
        z = self.prior.sample_z(mu, logvar)
        x_prime = self.decoder.decode(z)
        return x_prime, mu, logvar

    def print_model_info(self):
        for par in self.parameters():
            logger.debug(len(par))


if __name__=="__main__":
    logger.debug("Testing Model Setup") 
    model=DiVAE()
    logger.debug("Success")
    pass