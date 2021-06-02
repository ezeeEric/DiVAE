"""
Gumbel reparameterization trick Module

Author: Abhi (abhishek@myumanitoba.ca)
"""
import torch

class GumbelMod(torch.nn.Module):
    
    def __init__(self):
        super(GumbelMod, self).__init__()
        self.activation_fct = torch.nn.Sigmoid()
        
    def forward(self, logits, beta):
        """
        Gumbel reparameterization trick
        """
        rho = torch.rand(logits.size(), device=logits.device)
        logits_gumbel = logits + torch.log(rho) - torch.log(1 - rho)
        return self.activation_fct(logits_gumbel * beta)