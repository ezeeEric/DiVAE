"""
DVAE++ PyTorch

Author: Abhi (abhishek@myumanitoba.ca)
"""

# PyTorch imports
import torch

# DiVAE imports
from models.autoencoders.discreteVAE import DiVAE
from models.priors.rbm import RBM

from utils.distributions import Bernoulli

from networks.hiEncoder import HierarchicalEncoder
from networks.basicDecoder import BasicDecoder


class DiVAEPP(DiVAE):
    
    def __init__(self, **kwargs):
        super(DiVAE, self).__init__(**kwargs)
        self._model_type = "DiVAEPP"
        
    def _create_encoder(self):
        """
        Overrides _create_encoder in discreteVAE.py
        """
        logger.debug("ERROR _create_encoder dummy implementation")
        return HierarchicalEncoder(
            input_dimension=self._flat_input_size,
            n_latent_hierarchy_lvls=self.n_latent_hierarchy_lvls,
            n_latent_nodes=self.n_latent_nodes,
            n_encoder_layer_nodes=self.n_encoder_layer_nodes,
            n_encoder_layers=self.n_encoder_layers,
            skip_latent_layer=False,
            smoother="MixtureExp",
            cfg=self._config)
        
    def loss(self, input_data, fwd_out):
        """
        Overrides loss in discreteVAE.py
        """
        logger.debug("loss")
        
        kl_per_sample = self.kld()
    
        
        
    
            
        
        
            
        
            
        
    
    


