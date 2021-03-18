"""
Mixture of Exponential Module

Author: Abhi (abhishek@myumanitoba.ca)
"""
import torch

class MixtureExpMod(torch.nn.Module):
    """
    nn.Module implementation of mixture of overlapping 
    exponential distributions from DVAE++
    """
    def __init__(self):
        super(MixtureExpMod, self).__init__()
        
    def forward(self, logits, beta):
        """
        - ICDF of mixture of exponential distributions (Eq. 3, DVAE++)
        Returns:
            zeta: approximate post samples
        """
        q = torch.sigmoid(logits)
        q = torch.clamp(q, min=1e-7, max=1.-1e-7)
        
        rho = torch.rand(q.size())
        b = (rho + torch.exp(-beta)*(q - rho))/((1. - q) - 1.)
        c = -(q*torch.exp(-beta))/(1. - q)
        
        zeta = (-1./beta)*(torch.log((-b + torch.sqrt(b**2 - 4*c))/2.))
        return zeta