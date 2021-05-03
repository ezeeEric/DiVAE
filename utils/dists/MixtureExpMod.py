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
        Q = torch.sigmoid(logits)
        Q = torch.clamp(Q, min=1e-5, max=1.-1e-5)
        
        rho = torch.rand(Q.size(), device=Q.device)
        #B = (rho + torch.exp(-beta)*(Q - rho))/((1. - Q) - 1.)
        B = ((rho + torch.exp(-beta)*(Q - rho))/(1. - Q)) - 1.
        C = -(Q*torch.exp(-beta))/(1. - Q)
        M = (-B + torch.sqrt(B**2 - 4*C))/2.
        zeta = (-1./beta)*(torch.log(M))
        return zeta