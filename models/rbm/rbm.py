
"""
PyTorch implementation of a restricted Boltzmann machine
"""

import numpy as np
import torch

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

from DiVAE import logging
logger = logging.getLogger(__name__)

from models.rbm.rbmBase import RBMBase

class RBM(RBMBase):
    def __init__(self, **kwargs):
        super(RBM, self).__init__(**kwargs)
        # random weights and biases for all layers
        # weights between visible and hidden nodes. 784x128 (that is 28x28 input
        #size, 128 arbitrary choice)
        # if requires_grad=False : we calculate the weight update ourselves, not
        # through backpropagation
        require_grad=True
        #arbitrarily scaled by 0.01 
        self.weights = nn.Parameter(torch.randn(self._n_visible, self._n_hidden) * 0.01, requires_grad=require_grad)
        #all biases initialised to 0.5
        self.visible_bias = nn.Parameter(torch.ones(self._n_visible) * 0.5, requires_grad=require_grad)
        #applying a 0 bias to the hidden nodes
        self.hidden_bias = nn.Parameter(torch.zeros(self._n_hidden), requires_grad=require_grad)

    def get_logZ_value(self):
        #TODO include calculation of this value
        # this hardcoded number is taken from Figure 10 Rolfe
        return 33.65
    
    def energy(self, post_samples):
        """Energy Computation for RBM
        vis*b_vis+hid*b_hid+vis*w*hid
        Takes posterior samples as input
        """
        post_samples_concat=torch.cat(post_samples,dim=1)
        n_split=post_samples_concat.size()[1]//2
        post_samples_left,post_samples_right=torch.split(post_samples_concat,split_size_or_sections=int(n_split),dim=1)

        v=post_samples_left
        h=post_samples_right

        e_vis=torch.matmul(v,self.visible_bias)
        e_hid=torch.matmul(h,self.hidden_bias)
        e_mix=torch.sum(torch.matmul(v,self.weights)*h,axis=1)
        energy=-e_vis-e_hid-e_mix
        return energy

    def cross_entropy(self,post_samples):
        """For the case in Discrete VAE paper, this is simply the log prob (verify!)"""
        return -self.log_prob(post_samples)

    def log_prob(self, samples):
        """log(exp(-E(z))/Z) = -E(z) - log(Z)"""
        return -self.energy(samples)-self.get_logZ_value()