"""
Mixture of Exponential Distribution

Author: Abhi (abhishek@myumanitoba.ca)
"""
import torch
from torch import nn
from torch import log, exp
from torch.distributions.bernoulli import Bernoulli

class MixtureExp(Bernoulli):
    """
    Mixture of overlapping exponential distributions from DVAE++
    """
    def __init__(self, logits=None, beta=None):
        super(MixtureExp, self).__init__(logits=logits)
        self.beta = beta
    
    def log_pdf(self, zeta):
        """
        - Compute log(r(zeta|z = 0)), Note log(r(zeta|z = 1)) = log_pdf(self, 1-zeta)
        Args:
            zeta: approximate post samples
        
        Returns:
            log(r(zeta|z = 0))
        """
        return log(self.beta) - (self.beta * zeta) - log(1 - exp(-self.beta))
    
    def log_ratio(self, zeta):
        """
        - Compute log_ratio needed for gradients of KL (presented in DVAE++).
        Args:
            zeta: approximate post samples
        Returns:
            log_ratio: log r(\zeta|z=1) - log r(\zeta|z=0) 
        """
        log_pdf_0 = self.log_pdf(zeta)
        log_pdf_1 = self.log_pdf(1.- zeta)
        log_ratio = log_pdf_1 - log_pdf_0
        return log_ratio
        