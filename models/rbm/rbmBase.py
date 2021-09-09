
"""
Restricted Boltzmann Machine Abstract Base Class
"""

import numpy as np
import torch

from torch import nn
from torch.distributions import Distribution, Normal, Uniform

from DiVAE import logging
logger = logging.getLogger(__name__)

class RBMBase(nn.Module):
    def __init__(self, n_visible=2, n_hidden=2, weights=None, vis_bias=None, hid_bias=None, **kwargs):
        super(RBMBase, self).__init__(**kwargs)

        self._n_visible=n_visible
        self._n_hidden=n_hidden

        self._weights = weights
        self._visible_bias = vis_bias
        self._hidden_bias = hid_bias

    @property
    def n_visible(self):
        return self._n_visible

    @n_visible.setter
    def n_visible(self, n_vis):
        self._n_visible=n_vis

    @property
    def n_hidden(self):
        return self._n_hidden

    @n_hidden.setter
    def n_hidden(self, n_hid):
        self._n_hidden=n_hid

    @property
    def visible_bias(self):
        return self._visible_bias
    
    @visible_bias.setter
    def visible_bias(self,vis_bias):
        self._visible_bias=vis_bias
    
    @property
    def hidden_bias(self):
        return self._hidden_bias

    @hidden_bias.setter
    def hidden_bias(self, hid_bias):
        self._hidden_bias=hid_bias

    @property
    def weights(self):
        return self._weights
    
    @weights.setter
    def weights(self,w):
        self._weights=w

    def __repr__(self):
        outString="RBM: n_vis={0}, n_hid={1}".format(self._n_visible,self._n_hidden)
        outString+="\n\t weights ({0}): \n\t {1}".format(self._weights.size(),self.weights)
        outString+="\n\t visbias ({0}): \n\t {1}".format(self.visible_bias.size(),self.visible_bias)
        outString+="\n\t hidbias ({0}): \n\t {1}".format(self.hidden_bias.size(),self.hidden_bias)

        return outString

    def get_logZ_value(self):
        """Calculate and return logarithm of partition function Z.
        """
        return NotImplementedError
    
    def energy(self, post_samples):
        """Energy Computation for RBM
        vis*b_vis+hid*b_hid+vis*w*hid
        Takes posterior samples as input
        """
        return NotImplementedError

    def cross_entropy(self,post_samples):
        """For the case in Discrete VAE paper, this is simply the log prob (verify!)"""
        return NotImplementedError

    def log_prob(self, samples):
        """log(exp(-E(z))/Z) = -E(z) - log(Z)"""
        return NotImplementedError
