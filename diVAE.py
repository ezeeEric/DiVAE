# -*- coding: utf-8 -*-
"""
Discrete Variational Autoencoder Class Structures

Author: Eric Drechsler (eric_drechsler@sfu.ca)

Based on work from Olivia di Matteo.
"""

#pyTorch: Open source ML library dev. mainly by Facebook's AI research lab
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from networks import Encoder,Decoder
from rbm import RBM

from copy import copy
import logging
logger = logging.getLogger(__name__)

torch.manual_seed(1)

class AEBase(nn.Module):
    def __init__(self, latent_dimensions=32, **kwargs):
        super(AEBase,self).__init__(**kwargs)
        self.type=None
        self._latent_dimensions=latent_dimensions

    def _create_encoder(self):
        raise NotImplementedError

    def _create_decoder(self):
        raise NotImplementedError
    
    def __repr__(self):
        parameter_string="\n".join([str(par) for par in self.__dict__.items()])
        return parameter_string
    
    def forward(self, x):
        raise NotImplementedError

    def print_model_info(self):
        for par in self.__dict__.items():
            logger.debug(par)

#AE implementation, base class for VAE and DiVAE
class AE(AEBase):

    def __init__(self, latent_dimensions=32, **kwargs):
        super(AE,self).__init__(**kwargs)
        self.type="AE"
        
        self._encoder_nodes=[(784,128),(128,self._latent_dimensions)]
        self._decoder_nodes=[(self._latent_dimensions,128),(128,784)]
                
        # TODO replace the above with a more elegant solution
        # self.networkStructures={
        #     'encoder':[784,128,32],
        #     'decoder':[32,128,784]
        # }

        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()
        
        #TODO which is the best loss function for AE? Research.
        # nn.BCELoss(x_true,x_recon)
        self._loss_fct= nn.functional.binary_cross_entropy

    def _create_encoder(self):
        logger.debug("_create_encoder")
        return Encoder(node_sequence=self._encoder_nodes, activation_fct=nn.ReLU())

    def _create_decoder(self):
        logger.debug("_create_decoder")
        return Decoder(node_sequence=self._decoder_nodes, activation_fct=nn.ReLU(), output_activation_fct=nn.Sigmoid())

    def forward(self, x):
        zeta = self.encoder.encode(x.view(-1, 784))
        x_recon = self.decoder.decode(zeta)
        return x_recon
    
    def loss(self, x_true, x_recon):
        return self._loss_fct(x_recon, x_true.view(-1, 784), reduction='sum')


#VAE implementation
class VAE(AEBase):
    def __init__(self, **kwargs):
        super(VAE, self).__init__()
        
        self.type="VAE"

        self._encoder_nodes=[(784,128)]
        self._reparamNodes=(128,self._latent_dimensions)   
        self._decoder_nodes=[(self._latent_dimensions,128),(128,784)]

        self._reparamLayers=nn.ModuleDict(
            {'mu':nn.Linear(self._reparamNodes[0],self._reparamNodes[1]),
             'var':nn.Linear(self._reparamNodes[0],self._reparamNodes[1])
             })
        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()

    def _create_encoder(self):
        logger.debug("_create_encoder")
        return Encoder(node_sequence=self._encoder_nodes, activation_fct=nn.ReLU())

    def _create_decoder(self):
        logger.debug("_create_decoder")
        return Decoder(node_sequence=self._decoder_nodes, activation_fct=nn.ReLU(), output_activation_fct=nn.Sigmoid())
        
    def reparameterize(self, mu, logvar):
        """ Sample from the normal distributions corres and return var * samples + mu
        """
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5 * logvar)
        
    def loss(self, x, x_recon, mu, logvar):
        logger.debug("loss")
        # Autoencoding term
        auto_loss = torch.nn.functional.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        
        # KL loss term assuming Gaussian-distributed latent variables
        kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return auto_loss - kl_loss
                            
    def forward(self, x):
        x_prime = self.encoder.encode(x.view(-1, 784))
        mu = self._reparamLayers['mu'](x_prime)
        logvar = self._reparamLayers['var'](x_prime)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder.decode(z)
        return x_recon, mu, logvar    

class DiVAE(AE):
    def __init__(self, **kwargs):
        super(DiVAE, self).__init__(**kwargs)
        self.type="DiVAE"

        self._latent_dimensions=latent_dimensions
        self._encoder_nodes=[(784,128),]
        self._reparamNodes=(128,latent_dimensions)  
        self._decoder_nodes=[(latent_dimensions,128),]
        self._outputNodes=(128,784)     


        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()
        self.prior=self._create_prior()
    
    def _create_encoder(self):
        logger.debug("_create_encoder")
        #TODO hacked
        node_sequence=self._encoder_nodes+[self._reparamNodes]
        return Encoder(node_sequence=node_sequence, activation_fct=nn.ReLU())

    def _create_decoder(self):
        logger.debug("_create_decoder")
        return Decoder(
            node_sequence=self._decoder_nodes,
            activation_fct=nn.ReLU(),
            output_nodes=self._outputNodes,
            output_activation_fct=nn.Sigmoid(),
            )

    def _create_prior(self):
        logger.debug("_create_prior")
        return RBM(n_visible=4,n_hidden=4)
   
    def lossDiVAE(self, x, x_recon, posterior_distribution,posterior_samples):
        logger.debug("lossDiVAE")
        #TODO
        hierarchical_posterior=posterior_distribution
        prior=self.prior
        auto_loss=0
        kl_loss=self.kl_divergence(hierarchical_posterior,prior)
        return auto_loss - kl_loss
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

        
        
        #encoder.hierarchical_posterior
        #decoder.reconstruct
        #createOutput
        #calculate KLD and loss
    
    def kl_divergence(self, posterior , prior):
        torchKLD=dist.kl.kl_divergence(posterior,prior)
        
        print(torchKLD)
        return torchKLD
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
        #TODO this should yield posterior distribution and samples
        #this now (200806) gives back "smoother" and samples from smoother. not
        #hierarchical yet.
        posterior_distribution, posterior_samples = self.encoder.hierarchical_posterior(x.view(-1, 784))
        print(posterior_samples[0])
        x_prime = self.decoder.decode_posterior_sample(posterior_samples[0])
        
        return x_prime, posterior_distribution, posterior_samples

if __name__=="__main__":
    logger.debug("Testing Model Setup") 
    model=VAE()
    print(model)
    # model=DiVAE()
    logger.debug("Success")
    pass