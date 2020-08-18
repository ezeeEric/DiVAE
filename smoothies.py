# -*- coding: utf-8 -*-

"""
smoothies.py

A set of classes representing the smoothing functions that are used as a bridge
from discrete to continuous variables when training a DVAE.

Author :  ODM, Dr. Dre
"""

import torch
from torch import zeros, ones
from torch.distributions import Distribution, Uniform

import numpy as np
import logging
logger = logging.getLogger(__name__)

class Smoother(Distribution):
    """ Abstract class for smoothing functions. 
    """
    def __init__(self, beta, **kwargs):
        super(Smoother, self).__init__(**kwargs)
        if beta <= 0:
            raise ValueError(f"Value of {beta} for beta is invalid. The beta of a smoothing \
                function must be strictly greater than 0.")

        self.beta = beta

        # All the smoothing functions we are working with pull samples from [0, 1] , so 
        # give every function a copy of a uniform distribution.
        self.uniform = Uniform(low=0, high=1)
        return

    def evaluate(self, zeta, z):
        """ Computes the value of :math:`r(\zeta)` given a value z. 

        This is evaluated as:

        .. math::
            r(\zeta) = r(\zeta | z = 1)^z r(\zeta | z = 0)&{(1-z)}
        """
        raise NotImplementedError

    def icdf(self, rho, q):
        """ 
            Evaluate the inverse CDF on the provided data.
            :param float rho: Samples from uniform distribution 
            :param float q: The posterior distribution (e.g. for a Bernoulli
                distribution, pr(z=1|x).
        """
        raise NotImplementedError

    def __repr__(self):
        return "\n".join([str(item) for item in self.__dict__.items()])

class SpikeAndExponentialSmoother(Smoother):
    """ 
    Spike-and-exponential smoother from the original DVAE paper of Rolfe.
    """
    def __init__(self, beta=3, **kwargs):
        super(SpikeAndExponentialSmoother, self).__init__(beta, **kwargs)

    def icdf(self, rho, q):
        zero_mask = zeros(q.size())
        ones=torch.ones(q.size())
        interior_log = ((rho+q-ones)/q)*(np.exp(self.beta)-1)+ones
        conditional_log = (1./self.beta)*torch.log(interior_log)
        x=torch.where(rho >= 1 - q, conditional_log, zero_mask)
        return x

    def entropy(self):
        """ FROM DWAVE
        Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        print("--- ","call FactorialBernoulliUtil::entropy()")
        # mu = tf.nn.sigmoid(self.logit_mu)
        # ent = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=mu)
        ent=0
        return ent

def visualiseSmoother(rho,q,samples):
    import matplotlib.pyplot as plt
    #TODO make this pretty - 3 plots showing rho,q,samples
    # samples=samples[torch.nonzero(samples,as_tuple=True)]
    samples=torch.flatten(samples)
    plt.hist(samples.detach().numpy(),bins=100)
    plt.show()
    # qnonZero=q[torch.nonzero(q,as_tuple=True)]
    # samples=samples[torch.nonzero(samples,as_tuple=True)]

if __name__=="__main__":
    logger.info("Testing DistUtils Setup") 
    smooth=SpikeAndExponentialSmoother()
    import pickle
    qprime=pickle.load(open("datasample.pkl","rb"))
    rho=torch.rand(qprime.size())
    sigmoid=torch.nn.Sigmoid()
    q=sigmoid(qprime)
    rho=torch.rand(q.size())
    samples=smooth.icdf(rho,q)
    visualiseSmoother(rho,q,samples)
    logger.info("Success")
    pass