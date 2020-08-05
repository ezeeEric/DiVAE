# -*- coding: utf-8 -*-

"""
smoothers.py

A set of classes representing the smoothing functions that are used as a bridge
from discrete to continuous variables when training a DVAE.

Author :  ODM
"""

import torch
from torch import zeros, ones
from torch.distributions import Distribution, Uniform

import numpy as np

class Smoother(Distribution):
    """ Abstract class for smoothing functions. 
    """
    def __init__(self, beta):
        if beta <= 0:
            raise ValueError(f"Value of {beta} for beta is invalid. The beta of a smoothing \
                function must be strictly greater than 0.")

        self.beta = beta

        # All the smoothing functions we are working with pull samples from [0, 1] , so 
        # give every function a copy of a uniform distribution.
        self.u_dist = Uniform(low=0, high=1)
        return

    def evaluate(self, zeta, z):
        """ Computes the value of :math:`r(\zeta)` given a value z. 

        This is evaluated as:

        .. math::
            r(\zeta) = r(\zeta | z = 1)^z r(\zeta | z = 0)&{(1-z)}
        """
        pass

    def icdf(self, rho, q):
        """ 
            Evaluate the inverse CDF on the provided data.
            :param float rho: Samples from uniform distribution 
            :param float q: The posterior distribution (e.g. for a Bernoulli
                distribution, pr(z=1|x).
        """
        pass


class SymmetricSmoother(Smoother):
    """ The smoothing function introduced in the DVAE++ paper arXiv:1802.04920.
        
        :param float beta: A positive number. An inverse temperature that defines 
            the sharpness of the step in the distribution. Higher beta yields a
            sharper step. 
    """
    def __init__(self, beta=3):
        super(SymmetricSmoother, self).__init__(beta)
        self.z_beta = (1 - np.exp(-self.beta)) / self.beta
       
    def r_z0(self, zeta):
        return torch.exp(-self.beta * zeta) / self.z_beta
    
    def r_z1(self, zeta):
        return torch.exp(self.beta * (zeta - 1)) / self.z_beta

    def evaluate(self, zeta, z):
        r0 = self.r_z0(zeta)
        r1 = self.r_z1(zeta)
        return torch.pow(r1, z) * torch.pow(r0, 1 - z)

    def icdf(self, rho, q):
        # Variable names according to Eq. (3)
        b = (rho + np.exp(-self.beta) * (q - rho)) / (1 - q) - 1
        c = - q * np.exp(-self.beta) / (1 - q)
        quadratic_term = 0.5 * (-b + torch.sqrt(b*b - 4*c))
    
        return (-1/self.beta) * torch.log(quadratic_term)

class SpikeAndExponentialSmoother(Smoother):
    """ Spike-and-exponential smoother from the original DVAE paper of Rolfe.
    """
    def __init__(self, beta=3):
        super(SpikeAndExponentialSmoother, self).__init__(beta)

    def icdf(self, rho, q):
        zero_mask = zeros(q.size())

        interior_log = ((rho + q - 1) / q) * (np.exp(self.beta) - 1) + 1
        conditional_log = (1./self.beta) * torch.log(interior_log)

        return torch.where(rho >= 1 - q, conditional_log, zero_mask)


class MixtureOfRampsSmoother(Smoother):
    """ Appendix D1 of Rolfe.
    """
    def __init__(self, beta):
        super(MixtureOfRampsSmoother, self).__init__(beta)

    def icdf(self, rho, q):
        # See Eq. (20)
        numerator_root = (q - 1) * (q - 1) + (2 * q - 1) * rho
        numerator = q - 1 + torch.sqrt(numerator_root)

        denominator = 2 * q - 1

        conditional_value = numerator / denominator

        return torch.where(q != 0.5, conditional_value, rho)


class SpikeAndSlab(Smoother):
    """ Appendix D2 of Rolfe.
    """
    def __init__(self, beta):
        super(SpikeAndSlab, self).__init__(beta)

    def icdf(self, rho, q):
        zero_mask = zeros(q.size())
        conditional_value = 1 + (rho - 1)/q

        return torch.where(rho >= 1 - q, conditional_value, zero_mask)
