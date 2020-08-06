# -*- coding: utf-8 -*-
"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

#pyTorch: Open source ML library dev. mainly by Facebook's AI research lab
import torch
import torch.nn as nn

from smoothies import SpikeAndExponentialSmoother
from copy import copy
import logging
logger = logging.getLogger(__name__)

#Base Class
class Network(nn.Module):
    def __init__(self, node_sequence=None, create_module_list=True,**kwargs):
        super(Network, self).__init__(**kwargs)
        self._layers=nn.ModuleList([]) if create_module_list else None
        self._node_sequence=node_sequence
        if self._node_sequence and create_module_list:
            self._create_network()
        pass
    
    def _create_network(self):        
        for node in self._node_sequence:
            self._layers.append(
                nn.Linear(node[0],node[1])
            )
        return

class Encoder(Network):
    def __init__(self,activation_fct=None,smoothing_distribution=NotImplemented,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self._activation_fct=activation_fct
        self.smoothing_distribution=SpikeAndExponentialSmoother(beta=4)
        pass
    
    def encode(self, x):
        logger.debug("encode")
        for layer in self._layers:
            x=self._activation_fct(layer(x))
        return x
    
    def hierarchical_posterior(self,x, is_training=True):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to num_latent_layers and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent units in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        logger.debug("Encoder::hierarchical_posterior")
        print("--- ","start SimpleEncoder::hierarchical_posterior()")
        posterior = []
        post_samples = []
        #TODO switched off hierarchy for now.
        self.num_latent_layers=1
        # import pickle
        for i in range(self.num_latent_layers):
            # network_input = tf.concat(axis=-1, values=[input] + post_samples)  # concat x, z0, z1, ...
            # network = self.nets[i]
            # param = network.build_network(network_input, is_training)                      # create network
            # In the evaluation, we will use Bernoulli instead of continuous relaxations.
           # if not is_training and self.dist_util in {MixtureNormal, Spike_and_Exp, MixtureGeneric}:
          #      posterior_dist = FactorialBernoulliUtil([param[0]])
          #  else:
            qprime=self.encode(x)
            # pickle.dump(qprime,open( "datasample.pkl", "wb" ))
            sigmoid=torch.nn.Sigmoid()
            q=sigmoid(qprime)
            #returns tensor of size n of random variables drawn from uniform
            #dist in [0,1)
            rho=torch.rand(q.size())
            posterior_dist = self.smoothing_distribution # init posterior dist.
            samples=posterior_dist.icdf(rho,q)
            posterior.append(posterior_dist)
            post_samples.append(samples)

        print("--- ","end SimpleEncoder::hierarchical_posterior()")
        return posterior, post_samples
    
class Decoder(Network):
    def __init__(self,activation_fct=None,output_nodes=None,output_activation_fct=None,**kwargs):
        super(Decoder, self).__init__(**kwargs) 
        #activation function to be used for layers before last
        self._activation_fct=activation_fct

        #last output layer treated separately, as it needs sigmoid activation        
        self._output_layer=nn.Linear(output_nodes[0],output_nodes[1]) if output_nodes else None
        self._output_activation_fct=output_activation_fct

    def decode(self, z):
        logger.debug("Decoder::decode")
        for layer in self._layers:
            z=self._activation_fct(layer(z))
    #TODO I.3 is this the way to handle last output? Check if the application of
    #Distutil stuff necessary
        last_layer=self._output_layer(z)
        x_prime = self._output_activation_fct(last_layer)                 
        return x_prime
    
    def decode_posterior_sample(self, zeta):
        logger.debug("Decoder::decode")  
        print(zeta.size())
        print(self._layers)
        for layer in self._layers:
            zeta=self._activation_fct(layer(zeta))
    #TODO I.3 is this the way to handle last output? Check if the application of
    #Distutil stuff necessary
        last_layer=self._output_layer(zeta)
        x_prime = self._output_activation_fct(last_layer)                 
        return x_prime

class Prior(Network):
    
    def __init__(self,**kwargs):
        super(Prior, self).__init__(create_module_list=False, **kwargs)

        if self._node_sequence:
            self._layers=nn.ModuleDict(
                {'mu':nn.Linear(self._node_sequence[0],self._node_sequence[1]),
                'var':nn.Linear(self._node_sequence[0],self._node_sequence[1])
                })
        #rbm_prior=RBM(self.latent_dimensions,self.latent_dimensions)
        # return rbm_prior
        # pass
#TODO remove
    def reparameterize(self,x):
        logger.debug("reparameterize")
        mu = self._layers['mu'](x)
        logvar = self._layers['var'](x)
        return mu, logvar

#TODO remove

    def sample_z(self, mu, logvar):
        """ Sample from the normal distributions corres and return var * samples + mu
        """
        logger.debug("sample_z")
        eps = torch.randn_like(mu)
        return mu + eps*torch.exp(0.5 * logvar)
        
if __name__=="__main__":
    logger.debug("Testing Networks")
    nw=Network()
    encoder=Encoder()
    logger.debug(encoder._layers)

    decoder=Decoder()
    logger.debug(decoder._layers)

    prior=Prior(create_module_list=True)
    logger.debug(prior._layers)
    logger.debug("Success")
    pass