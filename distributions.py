# -*- coding: utf-8 -*-

"""
distributions.py

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

def sigmoid_cross_entropy_with_logits(logits, labels):
        logger.debug("WARNING sigmoid_cross_entropy_with_logits preliminary")
        # this is the equivalent to the DWave code's
        # sigmoid_cross_entropy_with_logits(): return logits * labels +
        # tf.nn.softplus(-logits)
        #z- logits (=output)
        #x- labels (=input data?)
        # TODO this implentation follows sigmoid_cross_entropy_with_logits
        # EXACTLY like DWave implementation
        #TODO check https://discuss.pytorch.org/t/equivalent-of-tensorflows-sigmoid-cross-entropy-with-logits-in-pytorch/1985/13
        #https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        #max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #TODO check if this changes something in the training
        sp=torch.nn.Softplus()
        # return torch.max(x_recon,torch.zeros(x_recon.size()))-x_recon*x_true-sp(torch.abs(x_recon))
        return logits - logits * labels + sp(-logits)


class Bernoulli(Distribution):
    def __init__(self, logit=None,  beta=1,  **kwargs):
        super(Bernoulli, self).__init__(**kwargs)
        #this is the raw (no output fct) output data of the latent layer stored
        #in this distribution.
        assert logit is not None, 'Distributions must be initialised with logit'
        assert not beta<=0, 'beta larger 0'
        self.logits=logit
        self.beta = beta

    def reparameterise(self):
        #draw samples from bernoulli distribution with probability p=1-q
        #where q is logits and rho acts as sample 
        #returns 0/1
        q = torch.sigmoid(self.logits) # equals 1-probability
        rho = torch.rand(q.size())
        bernoulli_sample = torch.where(rho<q, torch.ones(q.size()), zeros(q.size()))
        return bernoulli_sample
    
    def entropy(self):
        """
        Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        x = torch.sigmoid(self.logits)
        entropy = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=x)
        return entropy

    def log_prob_per_var(self, samples):
        """
        Compute the log probability of samples under distribution of this object.
            - (x - x * z + log(1 + exp(-x))),  where x is logits, and z is samples.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_prob: a matrix of log_prob (num_samples * num_vars).
        """
        log_prob = - sigmoid_cross_entropy_with_logits(logits=self.logits, labels=samples)
        return log_prob

    def log_prob(self, samples):
        """
        Call log_prob_per_var() and then compute the sum of log_prob of all the variables.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_p: A vector of log_prob for each sample.
        """
        log_p = self.log_prob_per_var(samples)
        log_p = torch.sum(log_p, 1)
        print("TEST log_prob!")
        import sys
        sys.exit()
        return log_p

    def __repr__(self):
        return "\n".join([str(item) for item in self.__dict__.items()])

class SpikeAndExponentialSmoother(Bernoulli):
    """ 
    Spike-and-exponential smoother from the original DVAE paper of Rolfe.
    """
    def __init__(self, beta=3, **kwargs):
        super(SpikeAndExponentialSmoother, self).__init__(beta, **kwargs)

    def reparameterise(self):
        #this is the approximate posterior probability
        q = torch.sigmoid(self.logits)
        #this is a uniformly sampled random number
        rho=torch.rand(q.size())
        zero_mask = zeros(q.size())
        ones=torch.ones(q.size())
        interior_log = ((rho+q-ones)/q)*(np.exp(self.beta)-1)+ones
        conditional_log = (1./self.beta)*torch.log(interior_log)
        zeta=torch.where(rho >= 1 - q, conditional_log, zero_mask)
        return zeta

    def entropy(self,x):
        """ FROM DWAVE
        Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        #TODO is x here the same as logits in DWave code?
        logger.debug("ERROR entropy()")
        z  = torch.sigmoid(x)
        # sp = torch.nn.Softplus()
        ent=x-x*z+torch.log(1+torch.exp(-x))
        # ent=x-torch.matmul(x,z)+torch.log(1+torch.exp(-x))
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