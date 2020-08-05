# -*- coding: utf-8 -*-
"""
Autoencoders

Author: Eric Drechsler (eric_drechsler@sfu.ca)
"""

#pyTorch: Open source ML library dev. mainly by Facebook's AI research lab
import torch
import torch.nn as nn

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
    def __init__(self,activation_fct=None,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self._activation_fct=activation_fct
        pass
    
    def encode(self, x):
        logger.debug("encode")

        for layer in self._layers:
            x=self._activation_fct(layer(x))
        return x
          
class Decoder(Network):
    def __init__(self,activation_fct=None,output_nodes=None,output_activation_fct=None,**kwargs):
        super(Decoder, self).__init__(**kwargs) 
        #activation function to be used for layers before last
        self._activation_fct=activation_fct

        #last output layer treated separately, as it needs sigmoid activation        
        self._output_layer=nn.Linear(output_nodes[0],output_nodes[1]) if output_nodes else None
        self._output_activation_fct=output_activation_fct

    def decode(self, z):
        logger.debug("decode")
        for layer in self._layers:
            z=self._activation_fct(layer(z))
        last_layer=self._output_layer(z)
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

    def reparameterize(self,x):
        logger.debug("reparameterize")
        mu = self._layers['mu'](x)
        logvar = self._layers['var'](x)
        return mu, logvar

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