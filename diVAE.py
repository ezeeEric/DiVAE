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
import torch.distributions as dist
import numpy as np

from networks import Encoder,Decoder,Prior
from rbm import RBM

from copy import copy
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(1)

class DiVAE(nn.Module):
    def __init__(self, isVAE=False, latent_dimensions=32):
        super(DiVAE, self).__init__()
       
        self.isVAE = isVAE
        
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
        #TODO hacked
        node_sequence=self._encoderNodes if self.isVAE else self._encoderNodes+[self._reparamNodes]
        return Encoder(node_sequence=node_sequence, activation_fct=nn.ReLU())

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
        if self.isVAE:
            return Prior(node_sequence=self._reparamNodes)
        else:
            return RBM(n_visible=4,n_hidden=4,node_sequence=self._reparamNodes)
    
    def loss(self, x, x_recon, mu, logvar):
        logger.debug("loss")
        # Autoencoding term
        auto_loss = torch.nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
   
    def lossDiVAE(self, x, x_recon, posterior_distribution,posterior_samples):
        logger.debug("lossDiVAE")
            #encoder.hierarchical_posterior
            #decoder.reconstruct
            #createOutput - #TODO missing distribution creation
            
            # calculate KLD and loss
            # Autoencoding term
            # auto_loss = torch.nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
                    # expected log prob p(x| z)
        # total_kl = self.prior.kl_dist_from(posterior, post_samples, is_training)
        # cost = - output_dist.log_prob_per_var(input)
        # cost = tf.reduce_sum(cost, axis=1)
        # # weight decay loss
        # enc_wd_loss = self.encoder.get_weight_decay()
        # dec_wd_loss = self.decoder.get_weight_decay()
        # prior_wd_loss = self.prior.get_weight_decay() if isinstance(self.prior, RBM) else 0
        # wd_loss = enc_wd_loss + dec_wd_loss + prior_wd_loss
        #  neg_elbo_per_sample = kl_coeff * total_kl + cost

        # if k > 1:
        #     neg_elbo_per_sample = tf.reshape(neg_elbo_per_sample, [-1, k])
        #     neg_elbo_per_sample = tf.reduce_mean(neg_elbo_per_sample, axis=1)

        # neg_elbo = tf.reduce_mean(neg_elbo_per_sample, name='neg_elbo

            # KL loss term assuming Gaussian-distributed latent variables
            # this can be calculated analytically... vs the pytorch implementation...
            # kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

            #TODO
            hierarchical_posterior=posterior_distribution
            prior=self.prior
            kl_loss=kl_divergence(hierarchical_posterior,prior)
            return auto_loss - kl_loss
        
        #encoder.hierarchical_posterior
        #decoder.reconstruct
        #createOutput
        #calculate KLD and loss
    
    def kl_divergence(self, posterior , prior):
        tochKLD=dist.kl.kl_divergence(posterior,prior)
        # #TODO
        # print("Calling Dummy implementation of kl divergence")
        # means=[0.9,0.8]

        # p=dist.Bernoulli(torch.tensor([means[0]]))
        # q=dist.Bernoulli(torch.tensor([means[1]]))
        
        # return dist.kl.kl_divergence(p,q)

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
        if self.isVAE:
            mu, logvar = self.hierarchical_posterior(x.view(-1, 784))
            z = self.prior.sample_z(mu, logvar)
            x_prime = self.decoder.decode(z)
            return x_prime, mu, logvar
        else:
            #TODO this should yield posterior distribution and samples
            #this now (200806) gives back "smoother" and samples from smoother. not
            #hierarchical yet.
            posterior_distribution, posterior_samples = self.encoder.hierarchical_posterior(x.view(-1, 784))
            print(posterior_samples[0])
            x_prime = self.decoder.decode_posterior_sample(posterior_samples[0])
            
            return x_prime, posterior_distribution, posterior_samples

#from other code
        #  form the encoder for z
        # posterior, post_samples = self.encoder.hierarchical_posterior(encoder_input, is_training)

    # def print_model_info(self):
    #     for par in self.parameters():
    #         logger.debug(len(par))
    def print_model_info(self):
        for par in self.__dict__.items():
            logger.debug(par)

if __name__=="__main__":
    logger.debug("Testing Model Setup") 
    model=DiVAE()
    logger.debug("Success")
    pass