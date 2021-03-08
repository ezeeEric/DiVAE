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
        
    
            
        
        
            
        
            
        
    
    


